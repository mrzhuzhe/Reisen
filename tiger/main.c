#include <stdio.h>
#include "utils.h"
#include <stdlib.h>


int main(){
    printf(" tiger complier go go go\n");

    int* a = malloc(100 * sizeof (*a));
    a[0] = 123;
    printf(" %d\n", a[0]);
    return 0;
}