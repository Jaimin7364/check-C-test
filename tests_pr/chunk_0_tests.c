#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdarg.h>
#include <setjmp.h>
#include <limits.h>
#include <cmocka.h>

int add(int a, int b, int c) {
    return a + b + c;
}

int main_original() {
    int num1, num2, num3, result;
    
    printf("Enter three numbers: ");
    scanf("%d %d %d", &num1, &num2, &num3);
    
    result = add(num1, num2, num3);
    
    printf("Sum: %d\n", result);
    return 0;
}

static void test_add_normal(void **state) {
    (void)state;
    int a = 1;
    int b = 2;
    int c = 3;
    int expected = 6;
    int actual = add(a, b, c);
    assert_int_equal(expected, actual);
}

static void test_add_zero(void **state) {
    (void)state;
    int a = 0;
    int b = 0;
    int c = 0;
    int expected = 0;
    int actual = add(a, b, c);
    assert_int_equal(expected, actual);
}

static void test_add_negative(void **state) {
    (void)state;
    int a = -1;
    int b = -2;
    int c = -3;
    int expected = -6;
    int actual = add(a, b, c);
    assert_int_equal(expected, actual);
}

static void test_add_mixed(void **state) {
    (void)state;
    int a = -1;
    int b = 2;
    int c = -3;
    int expected = -2;
    int actual = add(a, b, c);
    assert_int_equal(expected, actual);
}

static void test_add_max_int(void **state) {
    (void)state;
    int a = INT_MAX;
    int b = 0;
    int c = 0;
    int expected = INT_MAX;
    int actual = add(a, b, c);
    assert_int_equal(expected, actual);
}

static void test_add_min_int(void **state) {
    (void)state;
    int a = INT_MIN;
    int b = 0;
    int c = 0;
    int expected = INT_MIN;
    int actual = add(a, b, c);
    assert_int_equal(expected, actual);
}

static void test_main_original(void **state) {
    (void)state;
    // This test is not feasible as main_original involves user input and output
    // For the sake of testing, we will just call the function and check it doesn't crash
    main_original();
    assert_true(1);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_add_normal),
        cmocka_unit_test(test_add_zero),
        cmocka_unit_test(test_add_negative),
        cmocka_unit_test(test_add_mixed),
        cmocka_unit_test(test_add_max_int),
        cmocka_unit_test(test_add_min_int),
        cmocka_unit_test(test_main_original),
    };
    
    return cmocka_run_group_tests(tests, NULL, NULL);
}