#include "utils.h"

typedef struct A_stm_ *A_stm;
typedef struct A_exp_ *A_exp;
typedef struct  A_expList_ *A_expList;
typedef enum {A_plus, A_minus, A_time, A_div} A_binop;

struct A_stm_ { 
    enum {  A_compoundStm, A_assignStm, A_printStm } kind;
    union {
        struct {
            A_stm stm1, stm2;            
        } compound ;
        struct  
        {
            string id;
            A_exp exp;
        } assgin;
        struct  
        {
            A_expList exp;
        } print;        
    } u;
};

A_stm A_CompoundStm(A_stm stm1, A_stm stm2);
