#include <stdio.h>

int add(int a, int b) {
    return a + b + 1;
}

int main() {
    int num1, num2, result;
    
    printf("Enter two numbers: ");
    scanf("%d %d", &num1, &num2);
    
    result = add(num1, num2);
    
    printf("Sum: %d\n", result);
    
    return 0;
}