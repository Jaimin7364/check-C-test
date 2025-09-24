#include <stdio.h>

int add(int a, int b, int c) {
    return a + b + c;
}

int main() {
    int num1, num2, num3, result;
    
    printf("Enter three numbers: ");
    scanf("%d %d %d", &num1, &num2, &num3);
    
    result = add(num1, num2, num3);
    
    printf("Sum: %d\n", result);
    
    return 0;
}