#include <stdio.h>
#include "utils.h"
#include <stdlib.h>
#include "utils.h"
#include "slp.h"

int main(){
    A_AssignStm("a", A_OpExp(A_NumExp(5), A_plus, A_NumExp(3)));
    return 0;
}