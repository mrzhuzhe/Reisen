#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"

void *checked_malloc(int len){
    void *p = malloc(len);
    if (!p) {
        fprintf(stderr, "\n Ran out of memory!\n");
        exit(1);
    }
    return p;
}