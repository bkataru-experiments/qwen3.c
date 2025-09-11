#include <stdio.h>
#include <string.h>

int main() {
    char str[] = "Hello World";
    char *result = strchr(str, 'o');

    if (result) {
        printf("Found 'o' at position: %ld\n", result - str);
    } else {
        printf("Character not found.\n");
    }

    return 0;
}
