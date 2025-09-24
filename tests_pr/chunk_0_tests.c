#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdarg.h>
#include <setjmp.h>
#include <limits.h>
#include <cmocka.h>

// Function under test
int mul2(int a, int b) {
    return a * b;
}

// Test function for mul2 with positive numbers
static void test_mul2_positive(void **state) {
    (void)state;
    int result = mul2(2, 3);
    assert_int_equal(result, 6);
}

// Test function for mul2 with negative numbers
static void test_mul2_negative(void **state) {
    (void)state;
    int result = mul2(-2, 3);
    assert_int_equal(result, -6);
}

// Test function for mul2 with zero
static void test_mul2_zero(void **state) {
    (void)state;
    int result = mul2(2, 0);
    assert_int_equal(result, 0);
}

// Test function for mul2 with large numbers
static void test_mul2_large(void **state) {
    (void)state;
    int result = mul2(INT_MAX, 2);
    assert_int_equal(result, INT_MIN + 2);
}

// Test function for mul2 with overflow
static void test_mul2_overflow(void **state) {
    (void)state;
    int result = mul2(INT_MAX, INT_MAX);
    assert_true(result < 0); // Overflow check
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_mul2_positive),
        cmocka_unit_test(test_mul2_negative),
        cmocka_unit_test(test_mul2_zero),
        cmocka_unit_test(test_mul2_large),
        cmocka_unit_test(test_mul2_overflow),
    };

    return cmocka_run_group_tests(tests, NULL, NULL);
}