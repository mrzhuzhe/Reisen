#include <stdio.h>
#include "utils.h"
#include <stdlib.h>
#include "utils.h"
#include "slp.h"

int main(){
    printf(" id %s \n" , (A_AssignStm("a", A_OpExp(A_NumExp(5), A_plus, A_NumExp(3))))->u.assgin.id);
    printf(" exp id %s \n" , (A_AssignStm("a", A_OpExp(A_NumExp(5), A_plus, A_NumExp(3))))->u.assgin.exp->u.id);
    printf(" exp op %d \n" , (A_AssignStm("a", A_OpExp(A_NumExp(5), A_plus, A_NumExp(3))))->u.assgin.exp->u.num);

    //printf(" compound stm1 id %s \n" , ((A_AssignStm("a", A_OpExp(A_NumExp(5), A_plus, A_NumExp(3))))->u.compound.stm1)->u.assgin.id);
    //printf(" exp id %s \n" , (A_AssignStm("a", A_OpExp(A_NumExp(5), A_plus, A_NumExp(3))))->u.assgin.exp->u.id);
    
    return 0;
}