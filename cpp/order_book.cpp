#include "order_book.hpp"
#include <stdexcept>
#include <iomanip>
#include <sstream>

namespace order_book {

// ---- constructor ------------------------------------------------------------

OrderBook::OrderBook(const std::string& symbol) : symbol_(symbol) {}

// ---- public interface -------------------------------------------------------

std::vector<Fill> OrderBook::submit(Order order) {
    ++order_count_;

    // for market orders: convert to aggressive limit
    // bid market order → limit at far above best ask (will match everything available)
    // ask market order → limit at far below best bid
    if (order.type == OType::MARKET) {
        if (order.side == Side::BID) {
            order.price = best_ask().value_or(0) + 100'000;
        } else {
            order.price = best_bid().value_or(std::numeric_limits<Price>::max()) - 100'000;
        }
        order.type = OType::LIMIT;  // now treated as a limit for matching logic
    }

    // store the order
    orders_.emplace(order.id, order);
    Order& stored = orders_.at(order.id);

    std::vector<Fill> fills;

    if (order.side == Side::BID) {
        fills = match_bid(stored);
    } else {
        fills = match_ask(stored);
    }

    // if the order has remaining quantity, add it to the passive book
    if (stored.remaining > 0 && stored.status != Status::CANCELLED) {
        add_to_book(stored);
    }

    return fills;
}

bool OrderBook::cancel(OrderId id) {
    auto it = orders_.find(id);
    if (it == orders_.end()) return false;

    Order& order = it->second;
    if (order.status == Status::FILLED || order.status == Status::CANCELLED) {
        return false;
    }

    order.status = Status::CANCELLED;
    order.remaining = 0;

    // remove from the appropriate price level
    if (order.side == Side::BID) {
        remove_from_level(bids_, order.price, id);
    } else {
        remove_from_level(asks_, order.price, id);
    }

    return true;
}

// ---- market data ------------------------------------------------------------

std::optional<Price> OrderBook::best_bid() const {
    if (bids_.empty()) return std::nullopt;
    return bids_.begin()->first;
}

std::optional<Price> OrderBook::best_ask() const {
    if (asks_.empty()) return std::nullopt;
    return asks_.begin()->first;
}

std::optional<Price> OrderBook::mid_price() const {
    auto bb = best_bid();
    auto ba = best_ask();
    if (!bb || !ba) return std::nullopt;
    return (*bb + *ba) / 2;
}

Price OrderBook::spread() const {
    auto bb = best_bid();
    auto ba = best_ask();
    if (!bb || !ba) return 0;
    return *ba - *bb;
}

Quantity OrderBook::bid_depth(Price price) const {
    auto it = bids_.find(price);
    if (it == bids_.end()) return 0;
    return it->second.total_qty;
}

Quantity OrderBook::ask_depth(Price price) const {
    auto it = asks_.find(price);
    if (it == asks_.end()) return 0;
    return it->second.total_qty;
}

void OrderBook::print_top(int levels) const {
    // collect ask levels (ascending price, print top N in reverse for display)
    std::vector<std::pair<Price, Quantity>> ask_levels;
    int count = 0;
    for (auto& [price, level] : asks_) {
        if (count++ >= levels) break;
        ask_levels.push_back({price, level.total_qty});
    }

    std::cout << "\n  ---- " << symbol_ << " order book ----\n";
    std::cout << std::setw(12) << "qty"
              << std::setw(10) << "price"
              << "  side\n";
    std::cout << std::string(32, '-') << "\n";

    for (auto it = ask_levels.rbegin(); it != ask_levels.rend(); ++it) {
        std::cout << std::setw(12) << it->second
                  << std::setw(10) << (it->first / 100.0)
                  << "  ASK\n";
    }

    auto mid = mid_price();
    if (mid) {
        std::cout << "  --- mid: " << std::fixed << std::setprecision(2)
                  << (*mid / 100.0) << " | spread: "
                  << (spread() / 100.0) << " ---\n";
    }

    count = 0;
    for (auto& [price, level] : bids_) {
        if (count++ >= levels) break;
        std::cout << std::setw(12) << level.total_qty
                  << std::setw(10) << (price / 100.0)
                  << "  BID\n";
    }
    std::cout << "  " << std::string(30, '-') << "\n";
}

// ---- matching logic ---------------------------------------------------------

std::vector<Fill> OrderBook::match_bid(Order& incoming) {
    /*
     * An incoming bid order matches against the ask side.
     * We walk the ask book from lowest ask upward while:
     *   - There is remaining quantity on the incoming order
     *   - The best ask price <= incoming bid price
     */
    std::vector<Fill> fills;

    while (incoming.remaining > 0 && !asks_.empty()) {
        auto& [best_ask_price, level] = *asks_.begin();

        if (best_ask_price > incoming.price) break;   // no match

        while (incoming.remaining > 0 && !level.orders.empty()) {
            Order* passive = level.orders.front();

            if (passive->remaining == 0 || passive->status != Status::OPEN) {
                level.orders.pop();
                continue;
            }

            Quantity exec_qty = std::min(incoming.remaining, passive->remaining);
            Fill fill = execute(incoming, *passive, exec_qty);
            fills.push_back(fill);
            ++fill_count_;

            level.total_qty -= exec_qty;
        }

        // remove empty price levels
        if (level.orders.empty() || level.total_qty == 0) {
            asks_.erase(asks_.begin());
        }
    }

    return fills;
}

std::vector<Fill> OrderBook::match_ask(Order& incoming) {
    /*
     * An incoming ask order matches against the bid side.
     * Walk the bid book from highest bid downward while:
     *   - There is remaining quantity on the incoming order
     *   - The best bid price >= incoming ask price
     */
    std::vector<Fill> fills;

    while (incoming.remaining > 0 && !bids_.empty()) {
        auto& [best_bid_price, level] = *bids_.begin();

        if (best_bid_price < incoming.price) break;   // no match

        while (incoming.remaining > 0 && !level.orders.empty()) {
            Order* passive = level.orders.front();

            if (passive->remaining == 0 || passive->status != Status::OPEN) {
                level.orders.pop();
                continue;
            }

            Quantity exec_qty = std::min(incoming.remaining, passive->remaining);
            Fill fill = execute(incoming, *passive, exec_qty);
            fills.push_back(fill);
            ++fill_count_;

            level.total_qty -= exec_qty;
        }

        if (level.orders.empty() || level.total_qty == 0) {
            bids_.erase(bids_.begin());
        }
    }

    return fills;
}

Fill OrderBook::execute(Order& aggressor, Order& passive, Quantity qty) {
    // execution price is always the passive (resting) order's price
    // this is the rule at most exchanges: the aggressor accepts the passive price
    Price exec_price = passive.price;

    aggressor.remaining -= qty;
    passive.remaining   -= qty;

    if (aggressor.remaining == 0) aggressor.status = Status::FILLED;
    else                          aggressor.status = Status::PARTIAL;

    if (passive.remaining == 0)   passive.status   = Status::FILLED;
    else                          passive.status    = Status::PARTIAL;

    using namespace std::chrono;
    Timestamp ts = duration_cast<nanoseconds>(
        system_clock::now().time_since_epoch()
    ).count();

    return Fill{
        aggressor.id,
        passive.id,
        exec_price,
        qty,
        ts,
        aggressor.side,
    };
}

// ---- book management --------------------------------------------------------

void OrderBook::add_to_book(Order& order) {
    if (order.side == Side::BID) {
        auto& level = bids_[order.price];
        level.price = order.price;
        level.total_qty += order.remaining;
        level.orders.push(&order);
    } else {
        auto& level = asks_[order.price];
        level.price = order.price;
        level.total_qty += order.remaining;
        level.orders.push(&order);
    }
    order.status = Status::OPEN;
}

void OrderBook::remove_from_level(
    std::map<Price, PriceLevel, std::greater<Price>>& side,
    Price price, OrderId id)
{
    auto it = side.find(price);
    if (it == side.end()) return;

    PriceLevel& level = it->second;
    // rebuild queue without the cancelled order
    std::queue<Order*> new_q;
    while (!level.orders.empty()) {
        Order* o = level.orders.front();
        level.orders.pop();
        if (o->id != id) {
            new_q.push(o);
        } else {
            level.total_qty -= o->remaining;
        }
    }
    level.orders = new_q;

    if (level.orders.empty()) side.erase(it);
}

void OrderBook::remove_from_level(
    std::map<Price, PriceLevel, std::less<Price>>& side,
    Price price, OrderId id)
{
    auto it = side.find(price);
    if (it == side.end()) return;

    PriceLevel& level = it->second;
    std::queue<Order*> new_q;
    while (!level.orders.empty()) {
        Order* o = level.orders.front();
        level.orders.pop();
        if (o->id != id) {
            new_q.push(o);
        } else {
            level.total_qty -= o->remaining;
        }
    }
    level.orders = new_q;

    if (level.orders.empty()) side.erase(it);
}

}  // namespace order_book
