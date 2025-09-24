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

int add2(int a, int b) {
    return a + b;
}

static void test_add_normal(void **state) {
    (void)state;
    int result = add(1, 2, 3);
    assert_int_equal(result, 6);
}

static void test_add_edge_cases(void **state) {
    (void)state;
    int result = add(INT_MAX, 0, 0);
    assert_int_equal(result, INT_MAX);
    result = add(INT_MIN, 0, 0);
    assert_int_equal(result, INT_MIN);
}

static void test_add2_normal(void **state) {
    (void)state;
    int result = add2(1, 2);
    assert_int_equal(result, 3);
}

static void test_add2_edge_cases(void **state) {
    (void)state;
    int result = add2(INT_MAX, 0);
    assert_int_equal(result, INT_MAX);
    result = add2(INT_MIN, 0);
    assert_int_equal(result, INT_MIN);
}

static void test_add2_overflow(void **state) {
    (void)state;
    int result = add2(INT_MAX, 1);
    assert_true(result < 0); // overflow
}

static void test_add2_underflow(void **state) {
    (void)state;
    int result = add2(INT_MIN, -1);
    assert_true(result > 0); // underflow
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_add_normal),
        cmocka_unit_test(test_add_edge_cases),
        cmocka_unit_test(test_add2_normal),
        cmocka_unit_test(test_add2_edge_cases),
        cmocka_unit_test(test_add2_overflow),
        cmocka_unit_test(test_add2_underflow),
    };

    return cmocka_run_group_tests(tests, NULL, NULL);
}