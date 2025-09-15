#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdarg.h>
#include <setjmp.h>
#include <limits.h>
#include <cmocka.h>

// Function under test
int add(int a, int b) {
    return a + b;
}

static void test_add_normal_operation(void **state) {
    (void)state;
    int result = add(5, 7);
    assert_int_equal(result, 12);
}

static void test_add_edge_cases(void **state) {
    (void)state;
    int result = add(INT_MAX, 1);
    assert_int_equal(result, INT_MAX + 1);
    result = add(INT_MIN, -1);
    assert_int_equal(result, INT_MIN - 1);
}

static void test_add_negative_numbers(void **state) {
    (void)state;
    int result = add(-5, -7);
    assert_int_equal(result, -12);
}

static void test_add_zero(void **state) {
    (void)state;
    int result = add(0, 0);
    assert_int_equal(result, 0);
    result = add(5, 0);
    assert_int_equal(result, 5);
    result = add(0, 7);
    assert_int_equal(result, 7);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_add_normal_operation),
        cmocka_unit_test(test_add_edge_cases),
        cmocka_unit_test(test_add_negative_numbers),
        cmocka_unit_test(test_add_zero),
    };
    
    return cmocka_run_group_tests(tests, NULL, NULL);
}