#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace vdb {

constexpr std::array<uint8_t, 8> BitPos{0b00000001, 0b00000010, 0b00000100,
                                        0b00001000, 0b00010000, 0b00100000,
                                        0b01000000, 0b10000000};
constexpr std::array<uint8_t, 8> BitMask{0b11111111, 0b11111110, 0b11111100,
                                         0b11111000, 0b11110000, 0b11100000,
                                         0b11000000, 0b10000000};

constexpr std::array<uint64_t, 64> BitPos64 = {
    0x0000000000000001ULL, 0x0000000000000002ULL, 0x0000000000000004ULL,
    0x0000000000000008ULL, 0x0000000000000010ULL, 0x0000000000000020ULL,
    0x0000000000000040ULL, 0x0000000000000080ULL, 0x0000000000000100ULL,
    0x0000000000000200ULL, 0x0000000000000400ULL, 0x0000000000000800ULL,
    0x0000000000001000ULL, 0x0000000000002000ULL, 0x0000000000004000ULL,
    0x0000000000008000ULL, 0x0000000000010000ULL, 0x0000000000020000ULL,
    0x0000000000040000ULL, 0x0000000000080000ULL, 0x0000000000100000ULL,
    0x0000000000200000ULL, 0x0000000000400000ULL, 0x0000000000800000ULL,
    0x0000000001000000ULL, 0x0000000002000000ULL, 0x0000000004000000ULL,
    0x0000000008000000ULL, 0x0000000010000000ULL, 0x0000000020000000ULL,
    0x0000000040000000ULL, 0x0000000080000000ULL, 0x0000000100000000ULL,
    0x0000000200000000ULL, 0x0000000400000000ULL, 0x0000000800000000ULL,
    0x0000001000000000ULL, 0x0000002000000000ULL, 0x0000004000000000ULL,
    0x0000008000000000ULL, 0x0000010000000000ULL, 0x0000020000000000ULL,
    0x0000040000000000ULL, 0x0000080000000000ULL, 0x0000100000000000ULL,
    0x0000200000000000ULL, 0x0000400000000000ULL, 0x0000800000000000ULL,
    0x0001000000000000ULL, 0x0002000000000000ULL, 0x0004000000000000ULL,
    0x0008000000000000ULL, 0x0010000000000000ULL, 0x0020000000000000ULL,
    0x0040000000000000ULL, 0x0080000000000000ULL, 0x0100000000000000ULL,
    0x0200000000000000ULL, 0x0400000000000000ULL, 0x0800000000000000ULL,
    0x1000000000000000ULL, 0x2000000000000000ULL, 0x4000000000000000ULL,
    0x8000000000000000ULL};

constexpr char kRS = '\u001e';
constexpr char kGS = '\u001d';

constexpr const char* kCRLF = "\r\n";

// Internal column constants
constexpr const char* kInternalColumnPrefix = "__";
constexpr const char* kDeletedFlagColumn = "__deleted_flag";

// Helper function to check if column name is internal
// Returns true if the name starts with "__" (2 or more underscores)
// Examples:
//   "__col" -> true (internal)
//   "___col" -> true (internal)
//   "____col" -> true (internal)
//   "col" -> false (not internal, non-underscore characters allowed)
//   "_col" -> false (not internal, single underscore allowed)
inline bool IsInternalColumn(std::string_view name) {
  // Check if name starts with at least 2 underscores
  return name.length() >= 2 && name.substr(0, 2) == kInternalColumnPrefix;
}

}  // namespace vdb
