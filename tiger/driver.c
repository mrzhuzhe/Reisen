#include <stdio.h>
#include "utils.h"
#include "errormsg.h"
#include "tokens.h"

YYSTYPE yylval;

int yylex(void);

string toknames[] = {
    "ID", "STRING", "INT", "COMMA"
};