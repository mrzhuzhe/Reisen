#include "stdio.h"
#define U(x) ((x)&0377)
#define NLSTATE yyprevious=YYNEWLINE
#define BEGIN yybgin = yysvec+1
#define INITIAL 0
#define YYLERR yysvec
#define YYSTATE (yyestate-yysvec-1)