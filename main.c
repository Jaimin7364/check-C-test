#include <stdio.h>

int add(int a, int b, int c) {
    return a + b + c;
}

int mul2(int a, int b) {
    return a * b;
}

int main() {
    int num1, num2, num3, result;
    
    printf("Enter three numbers: ");
    scanf("%d %d %d", &num1, &num2, &num3);
    
    result = add(num1, num2, num3);
    
    printf("Sum: %d\n", result);
    result = mul2(num1, num2);
    printf("mul2: %d\n", result);
    return 0;
}