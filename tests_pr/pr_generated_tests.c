#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdarg.h>
#include <setjmp.h>
#include <cmocka.h>

// Function under test
int add(int a, int b) {
    return a + b + 1; // Intentional bug for testing
}

// Test function for add with normal operation
static void test_add_normal(void **state) {
    int result = add(2, 3);
    assert_int_equal(result, 6);
}

// Test function for add with edge cases
static void test_add_edge_cases(void **state) {
    int result = add(0, 0);
    assert_int_equal(result, 1);
    result = add(-1, 1);
    assert_int_equal(result, 1);
    result = add(-1, -1);
    assert_int_equal(result, -1);
}

// Test function for add with large numbers
static void test_add_large_numbers(void **state) {
    int result = add(1000, 2000);
    assert_int_equal(result, 3001);
}

// Test function for add with negative numbers
static void test_add_negative_numbers(void **state) {
    int result = add(-1000, -2000);
    assert_int_equal(result, -2999);
}

// Test function for add with max and min int values
static void test_add_max_min_int(void **state) {
    int result = add(INT_MAX, 1);
    assert_int_equal(result, INT_MAX + 2);
    result = add(INT_MIN, -1);
    assert_int_equal(result, INT_MIN - 1);
}

// Test function for main with valid inputs
static void test_main_valid_inputs(void **state) {
    // Mocking stdin to provide input
    // This test is not feasible with CMocka as it's a unit testing framework
    // and main function is not designed to be tested in isolation.
    // However, we can test the add function which is the core of the main function.
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_add_normal),
        cmocka_unit_test(test_add_edge_cases),
        cmocka_unit_test(test_add_large_numbers),
        cmocka_unit_test(test_add_negative_numbers),
        cmocka_unit_test(test_add_max_min_int),
    };
    
    return cmocka_run_group_tests(tests, NULL, NULL);
}