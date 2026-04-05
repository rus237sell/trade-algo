#include "order_book.hpp"
#include <iostream>
#include <cassert>

/*
 * Example usage and basic test of the order matching engine.
 *
 * Compile:
 *   g++ -std=c++17 -O2 -o order_book main.cpp order_book.cpp
 *
 * Run:
 *   ./order_book
 */

using namespace order_book;

static uint64_t next_id = 1;
static uint64_t now_ns() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
}

Order make_limit(Side side, Price price, Quantity qty, const std::string& sym) {
    return Order(next_id++, side, OType::LIMIT, price, qty, now_ns(), sym);
}

Order make_market(Side side, Quantity qty, const std::string& sym) {
    return Order(next_id++, side, OType::MARKET, 0, qty, now_ns(), sym);
}

void print_fill(const Fill& f) {
    std::cout << "  FILL: aggressive=" << f.aggressive_id
              << " passive=" << f.passive_id
              << " price=" << std::fixed << std::setprecision(2) << (f.fill_price / 100.0)
              << " qty=" << f.fill_qty
              << " side=" << (f.aggressor_side == Side::BID ? "BUY" : "SELL")
              << "\n";
}

int main() {
    std::cout << "=== order matching engine demo ===\n\n";

    OrderBook book("AAPL");

    // --- build a resting order book ---
    std::cout << "[1] seeding passive orders...\n";

    // bids at 14990, 14980, 14970 (in ticks, representing $149.90, $149.80, $149.70)
    book.submit(make_limit(Side::BID, 14990, 100, "AAPL"));
    book.submit(make_limit(Side::BID, 14990, 200, "AAPL"));   // same price, queued behind
    book.submit(make_limit(Side::BID, 14980, 300, "AAPL"));
    book.submit(make_limit(Side::BID, 14970, 500, "AAPL"));

    // asks at 15000, 15010, 15020
    book.submit(make_limit(Side::ASK, 15000, 150, "AAPL"));
    book.submit(make_limit(Side::ASK, 15010, 250, "AAPL"));
    book.submit(make_limit(Side::ASK, 15020, 400, "AAPL"));

    book.print_top(3);

    auto bb = book.best_bid();
    auto ba = book.best_ask();
    assert(bb.has_value() && *bb == 14990);
    assert(ba.has_value() && *ba == 15000);
    assert(book.spread() == 10);   // 10 ticks = $0.10 spread

    std::cout << "\n[2] submitting aggressive buy limit — crosses the spread...\n";
    // this bid at 15000 should match against the resting ask at 15000
    auto fills = book.submit(make_limit(Side::BID, 15000, 100, "AAPL"));
    for (const auto& f : fills) print_fill(f);
    assert(fills.size() == 1);
    assert(fills[0].fill_qty == 100);
    assert(fills[0].fill_price == 15000);

    book.print_top(3);

    std::cout << "\n[3] submitting market buy — clears multiple price levels...\n";
    // market buy for 300 shares: should hit the remaining 50@15000, 250@15010
    auto mkt_fills = book.submit(make_market(Side::BID, 300, "AAPL"));
    for (const auto& f : mkt_fills) print_fill(f);
    // expect two fills: 50 at 15000, 250 at 15010
    assert(mkt_fills.size() == 2);

    book.print_top(3);

    std::cout << "\n[4] cancel a resting bid...\n";
    // order id 2 was the second bid at 14990 (qty 200)
    bool cancelled = book.cancel(2);
    std::cout << "  cancel order 2: " << (cancelled ? "success" : "failed") << "\n";
    assert(cancelled);
    assert(book.bid_depth(14990) == 100);  // only the first 100-share bid remains

    book.print_top(3);

    std::cout << "\n[5] attempting to cancel an already-filled order...\n";
    // order id 5 (the ask at 15000, qty 150) was fully filled
    bool fill_cancel = book.cancel(5);
    std::cout << "  cancel filled order 5: " << (fill_cancel ? "success" : "correctly rejected") << "\n";
    assert(!fill_cancel);

    std::cout << "\n=== summary ===\n";
    std::cout << "  total orders submitted: " << book.total_orders() << "\n";
    std::cout << "  total fills generated:  " << book.total_fills() << "\n";
    std::cout << "  best bid:  " << (book.best_bid().has_value() ? book.best_bid().value() / 100.0 : 0.0) << "\n";
    std::cout << "  best ask:  " << (book.best_ask().has_value() ? book.best_ask().value() / 100.0 : 0.0) << "\n";

    std::cout << "\nall assertions passed.\n";
    return 0;
}
