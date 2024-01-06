#include <stdio.h>
#include "utils.h"
#include <stdlib.h>
#include "utils.h"
#include "slp.h"
#include "driver.h"

int main(int argc, char **argv){
    printf(" id %s \n" , (A_AssignStm("a", A_OpExp(A_NumExp(5), A_plus, A_NumExp(3))))->u.assgin.id);
    printf(" exp id %s \n" , (A_AssignStm("a", A_OpExp(A_NumExp(5), A_plus, A_NumExp(3))))->u.assgin.exp->u.id);
    printf(" exp op %d \n" , (A_AssignStm("a", A_OpExp(A_NumExp(5), A_plus, A_NumExp(3))))->u.assgin.exp->u.num);

    //printf(" compound stm1 id %s \n" , ((A_AssignStm("a", A_OpExp(A_NumExp(5), A_plus, A_NumExp(3))))->u.compound.stm1)->u.assgin.id);
    //printf(" exp id %s \n" , (A_AssignStm("a", A_OpExp(A_NumExp(5), A_plus, A_NumExp(3))))->u.assgin.exp->u.id);
    string fname;
    int tok;
    if (argc != 2) {
        fprintf(stderr, "usage: a.out filename\n"); 
        exit(1);
    }
    EM_reset(fname);
    for (;;){
        tok=yylex();
        if (tok==0) {
            break;
        }
        switch (tok)
        {
        case ID: 
        case STRING:
            printf("%10s %4d %s\n", tokname(tok), EM_tokPos, yylval.sval);
            break;
        case INT:
            printf("%10s %4d %d\n", tokname(tok), EM_tokPos, yylval.ival);
        default:
            printf("%10s %d\n", tokname(tok), EM_tokPos);
        }
    }

    return 0;
}