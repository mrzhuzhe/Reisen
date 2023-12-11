#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include "utils.h"
#include "errormsg.h"


bool anyErrors=FALSE;

static string fileName="";
static int lineNum=1;
int EM_tokPos=0;
extern FILE *yyin;

typedef struct intList { 
    int i; 
    struct intList *rest; 
} *IntList;

static IntList intList(int i, IntList rest) {
    IntList l = checked_malloc(sizeof *l);
    l->i = i;
    l->rest = rest;
    return l;
}