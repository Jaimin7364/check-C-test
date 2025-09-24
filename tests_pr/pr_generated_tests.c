#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdarg.h>
#include <setjmp.h>
#include <limits.h>
#include <cmocka.h>

// Function under test
int add(int a, int b, int c) {
    return a + b + c;
}

// Test function for add with normal operation
static void test_add_normal(void **state) {
    (void)state;
    int result = add(1, 2, 3);
    assert_int_equal(result, 6);
}

// Test function for add with negative numbers
static void test_add_negative(void **state) {
    (void)state;
    int result = add(-1, -2, -3);
    assert_int_equal(result, -6);
}

// Test function for add with zero
static void test_add_zero(void **state) {
    (void)state;
    int result = add(0, 0, 0);
    assert_int_equal(result, 0);
}

// Test function for add with large numbers
static void test_add_large(void **state) {
    (void)state;
    int result = add(INT_MAX, 1, 1);
    assert_int_equal(result, INT_MAX + 2);
}

// Test function for add with small numbers
static void test_add_small(void **state) {
    (void)state;
    int result = add(INT_MIN, -1, -1);
    assert_int_equal(result, INT_MIN - 2);
}

// Test suite setup
int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_add_normal),
        cmocka_unit_test(test_add_negative),
        cmocka_unit_test(test_add_zero),
        cmocka_unit_test(test_add_large),
        cmocka_unit_test(test_add_small),
    };
    
    return cmocka_run_group_tests(tests, NULL, NULL);
}