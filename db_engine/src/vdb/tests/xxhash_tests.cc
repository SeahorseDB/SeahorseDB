#include <array>
#include <string>
#include <vector>
#include <cstring>

#include <gtest/gtest.h>

#include "xxhash.hpp"

namespace vdb {

class XXHashTestSuite : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}

  // Test data
  const std::string test_string = "Hello, World!";
  const std::vector<int> test_vector = {1, 2, 3, 4, 5};
  const std::array<char, 5> test_array = {'a', 'b', 'c', 'd', 'e'};
  const char* test_buffer = "Test buffer data";
  const size_t test_buffer_size = 15;  // Length of "Test buffer data"
  const uint64_t test_seed = 42;
};

// Simple placeholder test to verify test framework works
TEST_F(XXHashTestSuite, TestFrameworkWorks) { EXPECT_TRUE(true); }

// Test that we can include the header in a real implementation
TEST_F(XXHashTestSuite, IncludeHeaderTest) {
  // This test just verifies that the test can compile with the xxhash header
  // The actual implementation would use the xxhash library
  EXPECT_TRUE(true);
}

// Test xxhash 32-bit
TEST_F(XXHashTestSuite, XXHash32Basic) {
  // Test with different input types
  xxh::hash32_t hash1 = xxh::xxhash<32>(test_string);
  xxh::hash32_t hash2 = xxh::xxhash<32>(test_vector);
  xxh::hash32_t hash3 = xxh::xxhash<32>(test_array);
  xxh::hash32_t hash4 =
      xxh::xxhash<32>(static_cast<const void*>(test_buffer), test_buffer_size);

  // Verify hashes are non-zero
  EXPECT_NE(hash1, 0);
  EXPECT_NE(hash2, 0);
  EXPECT_NE(hash3, 0);
  EXPECT_NE(hash4, 0);

  // Verify same input produces same hash
  EXPECT_EQ(hash1, xxh::xxhash<32>(test_string));
  EXPECT_EQ(hash2, xxh::xxhash<32>(test_vector));
  EXPECT_EQ(hash3, xxh::xxhash<32>(test_array));
  EXPECT_EQ(hash4, xxh::xxhash<32>(static_cast<const void*>(test_buffer),
                                   test_buffer_size));

  // Verify different inputs produce different hashes
  EXPECT_NE(hash1, hash2);
  EXPECT_NE(hash1, hash3);
  EXPECT_NE(hash1, hash4);
}

// Test xxhash 32-bit with seed
TEST_F(XXHashTestSuite, XXHash32WithSeed) {
  xxh::hash32_t hash1 = xxh::xxhash<32>(test_string);
  xxh::hash32_t hash2 = xxh::xxhash<32>(test_string, test_seed);

  // Verify seed changes the hash
  EXPECT_NE(hash1, hash2);
}

// Test xxhash 64-bit
TEST_F(XXHashTestSuite, XXHash64Basic) {
  // Test with different input types
  xxh::hash64_t hash1 = xxh::xxhash<64>(test_string);
  xxh::hash64_t hash2 = xxh::xxhash<64>(test_vector);
  xxh::hash64_t hash3 = xxh::xxhash<64>(test_array);
  xxh::hash64_t hash4 =
      xxh::xxhash<64>(static_cast<const void*>(test_buffer), test_buffer_size);

  // Verify hashes are non-zero
  EXPECT_NE(hash1, 0);
  EXPECT_NE(hash2, 0);
  EXPECT_NE(hash3, 0);
  EXPECT_NE(hash4, 0);

  // Verify same input produces same hash
  EXPECT_EQ(hash1, xxh::xxhash<64>(test_string));
  EXPECT_EQ(hash2, xxh::xxhash<64>(test_vector));
  EXPECT_EQ(hash3, xxh::xxhash<64>(test_array));
  EXPECT_EQ(hash4, xxh::xxhash<64>(static_cast<const void*>(test_buffer),
                                   test_buffer_size));

  // Verify different inputs produce different hashes
  EXPECT_NE(hash1, hash2);
  EXPECT_NE(hash1, hash3);
  EXPECT_NE(hash1, hash4);
}

// Test xxhash 64-bit with seed
TEST_F(XXHashTestSuite, XXHash64WithSeed) {
  xxh::hash64_t hash1 = xxh::xxhash<64>(test_string);
  xxh::hash64_t hash2 = xxh::xxhash<64>(test_string, test_seed);

  // Verify seed changes the hash
  EXPECT_NE(hash1, hash2);
}

// Test xxhash3 64-bit
TEST_F(XXHashTestSuite, XXHash3_64Basic) {
  // Test with different input types
  xxh::hash64_t hash1 = xxh::xxhash3<64>(test_string);
  xxh::hash64_t hash2 = xxh::xxhash3<64>(test_vector);
  xxh::hash64_t hash3 = xxh::xxhash3<64>(test_array);
  xxh::hash64_t hash4 =
      xxh::xxhash3<64>(static_cast<const void*>(test_buffer), test_buffer_size);

  // Verify hashes are non-zero
  EXPECT_NE(hash1, 0);
  EXPECT_NE(hash2, 0);
  EXPECT_NE(hash3, 0);
  EXPECT_NE(hash4, 0);

  // Verify same input produces same hash
  EXPECT_EQ(hash1, xxh::xxhash3<64>(test_string));
  EXPECT_EQ(hash2, xxh::xxhash3<64>(test_vector));
  EXPECT_EQ(hash3, xxh::xxhash3<64>(test_array));
  EXPECT_EQ(hash4, xxh::xxhash3<64>(static_cast<const void*>(test_buffer),
                                    test_buffer_size));

  // Verify different inputs produce different hashes
  EXPECT_NE(hash1, hash2);
  EXPECT_NE(hash1, hash3);
  EXPECT_NE(hash1, hash4);
}

// Test xxhash3 64-bit with seed
TEST_F(XXHashTestSuite, XXHash3_64WithSeed) {
  xxh::hash64_t hash1 = xxh::xxhash3<64>(test_string);
  xxh::hash64_t hash2 = xxh::xxhash3<64>(test_string, test_seed);

  // Verify seed changes the hash
  EXPECT_NE(hash1, hash2);
}

// Test xxhash3 128-bit
TEST_F(XXHashTestSuite, XXHash3_128Basic) {
  // Test with different input types
  xxh::hash128_t hash1 = xxh::xxhash3<128>(test_string);
  xxh::hash128_t hash2 = xxh::xxhash3<128>(test_vector);
  xxh::hash128_t hash3 = xxh::xxhash3<128>(test_array);
  xxh::hash128_t hash4 = xxh::xxhash3<128>(
      static_cast<const void*>(test_buffer), test_buffer_size);

  // Verify hashes are non-zero (both low64 and high64)
  EXPECT_NE(hash1.low64, 0);
  EXPECT_NE(hash1.high64, 0);
  EXPECT_NE(hash2.low64, 0);
  EXPECT_NE(hash2.high64, 0);
  EXPECT_NE(hash3.low64, 0);
  EXPECT_NE(hash3.high64, 0);
  EXPECT_NE(hash4.low64, 0);
  EXPECT_NE(hash4.high64, 0);

  // Verify same input produces same hash
  xxh::hash128_t hash1_dup = xxh::xxhash3<128>(test_string);
  EXPECT_EQ(hash1.low64, hash1_dup.low64);
  EXPECT_EQ(hash1.high64, hash1_dup.high64);

  // Verify different inputs produce different hashes
  EXPECT_NE(hash1.low64, hash2.low64);
  EXPECT_NE(hash1.high64, hash2.high64);
}

// Test xxhash3 128-bit with seed
TEST_F(XXHashTestSuite, XXHash3_128WithSeed) {
  xxh::hash128_t hash1 = xxh::xxhash3<128>(test_string);
  xxh::hash128_t hash2 = xxh::xxhash3<128>(test_string, test_seed);

  // Verify seed changes the hash
  EXPECT_NE(hash1.low64, hash2.low64);
  EXPECT_NE(hash1.high64, hash2.high64);
}

// Test xxhash streaming API for 32-bit
TEST_F(XXHashTestSuite, XXHash32Streaming) {
  // Create a streaming state
  xxh::hash_state32_t state1;

  // Update with different chunks of data
  state1.update(test_string.substr(0, 5));
  state1.update(test_string.substr(5));

  // Get the final hash
  xxh::hash32_t hash1 = state1.digest();

  // Compare with non-streaming version
  xxh::hash32_t hash2 = xxh::xxhash<32>(test_string);

  EXPECT_EQ(hash1, hash2);

  // Test with seed
  xxh::hash_state32_t state3(test_seed);
  state3.update(test_string);
  xxh::hash32_t hash3 = state3.digest();
  xxh::hash32_t hash4 = xxh::xxhash<32>(test_string, test_seed);

  EXPECT_EQ(hash3, hash4);
}

// Test xxhash streaming API for 64-bit
TEST_F(XXHashTestSuite, XXHash64Streaming) {
  // Create a streaming state
  xxh::hash_state64_t state1;

  // Update with different chunks of data
  state1.update(test_string.substr(0, 5));
  state1.update(test_string.substr(5));

  // Get the final hash
  xxh::hash64_t hash1 = state1.digest();

  // Compare with non-streaming version
  xxh::hash64_t hash2 = xxh::xxhash<64>(test_string);

  EXPECT_EQ(hash1, hash2);

  // Test with seed
  xxh::hash_state64_t state3(test_seed);
  state3.update(test_string);
  xxh::hash64_t hash3 = state3.digest();
  xxh::hash64_t hash4 = xxh::xxhash<64>(test_string, test_seed);

  EXPECT_EQ(hash3, hash4);
}

// Test xxhash3 streaming API for 64-bit
TEST_F(XXHashTestSuite, XXHash3_64Streaming) {
  // Create a streaming state
  xxh::hash3_state64_t state1;

  // Update with different chunks of data
  state1.update(test_string.substr(0, 5));
  state1.update(test_string.substr(5));

  // Get the final hash
  xxh::hash64_t hash1 = state1.digest();

  // Compare with non-streaming version
  xxh::hash64_t hash2 = xxh::xxhash3<64>(test_string);

  EXPECT_EQ(hash1, hash2);

  // Test with seed
  xxh::hash3_state64_t state3(test_seed);
  state3.update(test_string);
  xxh::hash64_t hash3 = state3.digest();
  xxh::hash64_t hash4 = xxh::xxhash3<64>(test_string, test_seed);

  EXPECT_EQ(hash3, hash4);
}

// Test xxhash3 streaming API for 128-bit
TEST_F(XXHashTestSuite, XXHash3_128Streaming) {
  // Create a streaming state
  xxh::hash3_state128_t state1;

  // Update with different chunks of data
  state1.update(test_string.substr(0, 5));
  state1.update(test_string.substr(5));

  // Get the final hash
  xxh::hash128_t hash1 = state1.digest();

  // Compare with non-streaming version
  xxh::hash128_t hash2 = xxh::xxhash3<128>(test_string);

  EXPECT_EQ(hash1.low64, hash2.low64);
  EXPECT_EQ(hash1.high64, hash2.high64);

  // Test with seed
  xxh::hash3_state128_t state3(test_seed);
  state3.update(test_string);
  xxh::hash128_t hash3 = state3.digest();
  xxh::hash128_t hash4 = xxh::xxhash3<128>(test_string, test_seed);

  EXPECT_EQ(hash3.low64, hash4.low64);
  EXPECT_EQ(hash3.high64, hash4.high64);
}

// Test custom secret for xxhash3
TEST_F(XXHashTestSuite, XXHash3CustomSecret) {
  std::array<uint8_t, 192> custom_secret;
  xxh::generate_secret(custom_secret.data(), custom_secret.size(), &test_seed,
                       sizeof(test_seed));

  // Test 64-bit xxhash3 with custom secret
  xxh::hash64_t hash1 = xxh::xxhash3<64>(test_string);
  xxh::hash64_t hash2 =
      xxh::xxhash3<64>(test_string, custom_secret.data(), custom_secret.size());

  // Should be different because we're using a custom secret
  EXPECT_NE(hash1, hash2);

  // Test 128-bit xxhash3 with custom secret
  xxh::hash128_t hash3 = xxh::xxhash3<128>(test_string);
  xxh::hash128_t hash4 = xxh::xxhash3<128>(test_string, custom_secret.data(),
                                           custom_secret.size());

  // Should be different because we're using a custom secret
  EXPECT_NE(hash3.low64, hash4.low64);
  EXPECT_NE(hash3.high64, hash4.high64);
}

// Test reset functionality in streaming APIs
TEST_F(XXHashTestSuite, StreamingReset) {
  // Test xxhash 32-bit reset
  xxh::hash_state32_t state32;
  state32.update(test_string);
  xxh::hash32_t hash1 = state32.digest();

  state32.reset();
  state32.update(test_string);
  xxh::hash32_t hash2 = state32.digest();

  EXPECT_EQ(hash1, hash2);

  // Test xxhash 64-bit reset
  xxh::hash_state64_t state64;
  state64.update(test_string);
  xxh::hash64_t hash3 = state64.digest();

  state64.reset();
  state64.update(test_string);
  xxh::hash64_t hash4 = state64.digest();

  EXPECT_EQ(hash3, hash4);

  // Test xxhash3 64-bit reset
  xxh::hash3_state64_t state3_64;
  state3_64.update(test_string);
  xxh::hash64_t hash5 = state3_64.digest();

  state3_64.reset();
  state3_64.update(test_string);
  xxh::hash64_t hash6 = state3_64.digest();

  EXPECT_EQ(hash5, hash6);

  // Test xxhash3 128-bit reset
  xxh::hash3_state128_t state3_128;
  state3_128.update(test_string);
  xxh::hash128_t hash7 = state3_128.digest();

  state3_128.reset();
  state3_128.update(test_string);
  xxh::hash128_t hash8 = state3_128.digest();

  EXPECT_EQ(hash7.low64, hash8.low64);
  EXPECT_EQ(hash7.high64, hash8.high64);
}

// Test canonical representation
TEST_F(XXHashTestSuite, CanonicalRepresentation) {
  // Test 32-bit canonical
  xxh::hash32_t hash32 = xxh::xxhash<32>(test_string);
  xxh::canonical32_t canon32(hash32);
  xxh::hash32_t hash32_from_canon = canon32.get_hash();
  EXPECT_EQ(hash32, hash32_from_canon);

  // Test 64-bit canonical
  xxh::hash64_t hash64 = xxh::xxhash<64>(test_string);
  xxh::canonical64_t canon64(hash64);
  xxh::hash64_t hash64_from_canon = canon64.get_hash();
  EXPECT_EQ(hash64, hash64_from_canon);

  // Test 128-bit canonical
  xxh::hash128_t hash128 = xxh::xxhash3<128>(test_string);
  xxh::canonical128_t canon128(hash128);
  xxh::hash128_t hash128_from_canon = canon128.get_hash();
  EXPECT_EQ(hash128.low64, hash128_from_canon.low64);
  EXPECT_EQ(hash128.high64, hash128_from_canon.high64);
}

// Test edge cases
TEST_F(XXHashTestSuite, EdgeCases) {
  // Empty input
  std::string empty_str = "";
  std::vector<int> empty_vec = {};

  // xxhash 32-bit with empty input
  xxh::hash32_t hash1 = xxh::xxhash<32>(empty_str);
  xxh::hash32_t hash2 = xxh::xxhash<32>(empty_vec);
  EXPECT_NE(hash1, 0);  // Should still produce a non-zero hash
  EXPECT_EQ(hash1,
            hash2);  // Empty inputs of different types should hash the same

  // xxhash 64-bit with empty input
  xxh::hash64_t hash3 = xxh::xxhash<64>(empty_str);
  xxh::hash64_t hash4 = xxh::xxhash<64>(empty_vec);
  EXPECT_NE(hash3, 0);
  EXPECT_EQ(hash3, hash4);

  // xxhash3 64-bit with empty input
  xxh::hash64_t hash5 = xxh::xxhash3<64>(empty_str);
  xxh::hash64_t hash6 = xxh::xxhash3<64>(empty_vec);
  EXPECT_NE(hash5, 0);
  EXPECT_EQ(hash5, hash6);

  // xxhash3 128-bit with empty input
  xxh::hash128_t hash7 = xxh::xxhash3<128>(empty_str);
  xxh::hash128_t hash8 = xxh::xxhash3<128>(empty_vec);
  EXPECT_NE(hash7.low64, 0);
  EXPECT_NE(hash7.high64, 0);
  EXPECT_EQ(hash7.low64, hash8.low64);
  EXPECT_EQ(hash7.high64, hash8.high64);

  // Single byte input
  std::string single_byte = "a";
  xxh::hash32_t hash9 = xxh::xxhash<32>(single_byte);
  xxh::hash64_t hash10 = xxh::xxhash<64>(single_byte);
  xxh::hash64_t hash11 = xxh::xxhash3<64>(single_byte);
  xxh::hash128_t hash12 = xxh::xxhash3<128>(single_byte);

  EXPECT_NE(hash9, 0);
  EXPECT_NE(hash10, 0);
  EXPECT_NE(hash11, 0);
  EXPECT_NE(hash12.low64, 0);
  EXPECT_NE(hash12.high64, 0);
}

// Test consistency across different input methods
TEST_F(XXHashTestSuite, InputConsistency) {
  // Create the same data in different formats
  const char* raw_data = "Consistent test data";
  size_t data_len = strlen(raw_data);
  std::string str_data(raw_data);
  std::vector<char> vec_data(raw_data, raw_data + data_len);

  // Test xxhash 32-bit
  xxh::hash32_t hash1 =
      xxh::xxhash<32>(static_cast<const void*>(raw_data), data_len);
  xxh::hash32_t hash2 = xxh::xxhash<32>(str_data);
  xxh::hash32_t hash3 = xxh::xxhash<32>(vec_data);
  xxh::hash32_t hash4 = xxh::xxhash<32>(vec_data.begin(), vec_data.end());

  EXPECT_EQ(hash1, hash2);
  EXPECT_EQ(hash1, hash3);
  EXPECT_EQ(hash1, hash4);

  // Test xxhash 64-bit
  xxh::hash64_t hash5 =
      xxh::xxhash<64>(static_cast<const void*>(raw_data), data_len);
  xxh::hash64_t hash6 = xxh::xxhash<64>(str_data);
  xxh::hash64_t hash7 = xxh::xxhash<64>(vec_data);
  xxh::hash64_t hash8 = xxh::xxhash<64>(vec_data.begin(), vec_data.end());

  EXPECT_EQ(hash5, hash6);
  EXPECT_EQ(hash5, hash7);
  EXPECT_EQ(hash5, hash8);

  // Test xxhash3 64-bit
  xxh::hash64_t hash9 =
      xxh::xxhash3<64>(static_cast<const void*>(raw_data), data_len);
  xxh::hash64_t hash10 = xxh::xxhash3<64>(str_data);
  xxh::hash64_t hash11 = xxh::xxhash3<64>(vec_data);
  xxh::hash64_t hash12 = xxh::xxhash3<64>(vec_data.begin(), vec_data.end());

  EXPECT_EQ(hash9, hash10);
  EXPECT_EQ(hash9, hash11);
  EXPECT_EQ(hash9, hash12);

  // Test xxhash3 128-bit
  xxh::hash128_t hash13 =
      xxh::xxhash3<128>(static_cast<const void*>(raw_data), data_len);
  xxh::hash128_t hash14 = xxh::xxhash3<128>(str_data);
  xxh::hash128_t hash15 = xxh::xxhash3<128>(vec_data);
  xxh::hash128_t hash16 = xxh::xxhash3<128>(vec_data.begin(), vec_data.end());

  EXPECT_EQ(hash13.low64, hash14.low64);
  EXPECT_EQ(hash13.high64, hash14.high64);
  EXPECT_EQ(hash13.low64, hash15.low64);
  EXPECT_EQ(hash13.high64, hash15.high64);
  EXPECT_EQ(hash13.low64, hash16.low64);
  EXPECT_EQ(hash13.high64, hash16.high64);
}

// TODO: Implement xxhash tests once the sse2neon.h dependency is resolved
// For ARM64 architecture, we need to ensure the sse2neon.h header is available
// as required by the xxhash library. This header provides SSE intrinsics for
// ARM NEON.

}  // namespace vdb
