#pragma once

#include <cstdint>
#include <string>
#include <map>
#include <queue>
#include <vector>
#include <chrono>
#include <functional>
#include <optional>
#include <iostream>

/*
 * Limit Order Book — Price-Time Priority Matching Engine
 *
 * This is a simplified but architecturally correct implementation of the
 * core component inside every exchange and ECN.
 *
 * Price-time priority means:
 *   1. Orders at better prices are matched first.
 *      For bids: higher price is better.
 *      For asks: lower price is better.
 *   2. At the same price, orders placed earlier are matched first (FIFO).
 *
 * Order types supported:
 *   LIMIT  — rests in the book at the specified price until filled or cancelled
 *   MARKET — executes immediately against the best available opposite-side price
 *            (converts to an aggressive limit at a safe worst-case price)
 *
 * Data structures:
 *   Bid side: std::map<price, queue<Order>> with reverse (descending) comparator
 *   Ask side: std::map<price, queue<Order>> with default (ascending) comparator
 *
 *   Map lookup is O(log N) for N distinct price levels.
 *   Within-level insertion and removal is O(1) (front of queue).
 *
 * In production systems this would be replaced with:
 *   - Lock-free intrusive linked lists per price level
 *   - Flat array indexed by price tick for O(1) level access
 *   - Separate thread-per-symbol with SPSC queues for order intake
 */

namespace order_book {

// ---- types ------------------------------------------------------------------

using OrderId   = uint64_t;
using Price     = int64_t;   // in integer ticks (e.g. price * 100 for 2dp precision)
using Quantity  = uint32_t;
using Timestamp = uint64_t;  // nanoseconds since epoch

enum class Side   { BID, ASK };
enum class OType  { LIMIT, MARKET };
enum class Status { OPEN, PARTIAL, FILLED, CANCELLED };

// ---- order ------------------------------------------------------------------

struct Order {
    OrderId   id;
    Side      side;
    OType     type;
    Price     price;       // 0 for market orders
    Quantity  quantity;    // original quantity
    Quantity  remaining;   // unfilled quantity
    Timestamp timestamp;
    Status    status;
    std::string symbol;

    Order(OrderId id_, Side side_, OType type_, Price price_,
          Quantity qty_, Timestamp ts_, const std::string& sym)
        : id(id_), side(side_), type(type_), price(price_),
          quantity(qty_), remaining(qty_), timestamp(ts_),
          status(Status::OPEN), symbol(sym) {}
};

// ---- fill report ------------------------------------------------------------

struct Fill {
    OrderId  aggressive_id;   // the incoming order that triggered the match
    OrderId  passive_id;      // the resting order that was hit
    Price    fill_price;      // price at which execution occurred
    Quantity fill_qty;        // quantity executed
    Timestamp timestamp;
    Side      aggressor_side;
};

// ---- price level ------------------------------------------------------------

struct PriceLevel {
    Price    price;
    Quantity total_qty;     // sum of remaining qty at this level
    std::queue<Order*> orders;  // FIFO queue of pointers to order storage

    PriceLevel() : price(0), total_qty(0) {}
    explicit PriceLevel(Price p) : price(p), total_qty(0) {}
};

// ---- order book -------------------------------------------------------------

class OrderBook {
public:
    explicit OrderBook(const std::string& symbol);

    // submit a new order; returns list of fills generated
    std::vector<Fill> submit(Order order);

    // cancel an existing order by id; returns true if cancelled
    bool cancel(OrderId id);

    // read-only market data
    std::optional<Price>    best_bid() const;
    std::optional<Price>    best_ask() const;
    std::optional<Price>    mid_price() const;
    Price                   spread() const;         // ask - bid in ticks
    Quantity                bid_depth(Price price) const;
    Quantity                ask_depth(Price price) const;
    void                    print_top(int levels = 5) const;

    // statistics
    uint64_t total_fills()  const { return fill_count_; }
    uint64_t total_orders() const { return order_count_; }

private:
    std::string symbol_;

    // bid side: descending price (highest bid first)
    std::map<Price, PriceLevel, std::greater<Price>> bids_;

    // ask side: ascending price (lowest ask first)
    std::map<Price, PriceLevel, std::less<Price>> asks_;

    // central order store: id -> Order
    std::map<OrderId, Order> orders_;

    uint64_t fill_count_  = 0;
    uint64_t order_count_ = 0;

    // matching logic
    std::vector<Fill> match_bid(Order& incoming);
    std::vector<Fill> match_ask(Order& incoming);
    Fill              execute(Order& aggressor, Order& passive, Quantity qty);

    void add_to_book(Order& order);
    void remove_from_level(std::map<Price, PriceLevel, std::greater<Price>>& side,
                           Price price, OrderId id);
    void remove_from_level(std::map<Price, PriceLevel, std::less<Price>>& side,
                           Price price, OrderId id);
};

}  // namespace order_book
