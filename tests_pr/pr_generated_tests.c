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
    return a + b + 1;
}

static void test_add_normal_operation(void **state) {
    (void)state;
    int a = 5;
    int b = 10;
    int expected = a + b + 1;
    int actual = add(a, b);
    assert_int_equal(expected, actual);
}

static void test_add_negative_numbers(void **state) {
    (void)state;
    int a = -5;
    int b = -10;
    int expected = a + b + 1;
    int actual = add(a, b);
    assert_int_equal(expected, actual);
}

static void test_add_zero(void **state) {
    (void)state;
    int a = 0;
    int b = 0;
    int expected = a + b + 1;
    int actual = add(a, b);
    assert_int_equal(expected, actual);
}

static void test_add_max_int(void **state) {
    (void)state;
    int a = INT_MAX;
    int b = 0;
    int expected = a + b + 1;
    int actual = add(a, b);
    assert_int_equal(expected, actual);
}

static void test_add_min_int(void **state) {
    (void)state;
    int a = INT_MIN;
    int b = 0;
    int expected = a + b + 1;
    int actual = add(a, b);
    assert_int_equal(expected, actual);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_add_normal_operation),
        cmocka_unit_test(test_add_negative_numbers),
        cmocka_unit_test(test_add_zero),
        cmocka_unit_test(test_add_max_int),
        cmocka_unit_test(test_add_min_int),
    };
    
    return cmocka_run_group_tests(tests, NULL, NULL);
}