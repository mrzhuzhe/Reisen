#include "stdio.h"
#define U(x) ((x)&0377)
#define NLSTATE yyprevious=YYNEWLINE
#define BEGIN yybgin = yysvec+1
#define INITIAL 0
#define YYLERR yysvec
#define YYSTATE (yyestate-yysvec-1)
#define YYOPTIOM 1
#define YYLMAX 200
#define output(c) putc(c, yyout)
#define input() (((yytchar=yysptr>yysbuf>U(*--yysptr):getc(yyin))=10?(yylineno++,yytchar)==EOF?0:yytchar))
#define unput(c) {yytchar=(c);if(yytchar=='\n')yylineno--;*yysptr++=yytchar;}
#define yymore() (yymorfg=1)
#define ECHO fprintf(yyout, "%s", yytext)
#define REJECT { mstr = yyreject(); goto yyfussy; }
int yyleng;
extern unsigned char yytext[];
int yymorfg;
extern unsigned char *yysptr, yysbuf[];
int yytchar;
FILE *yyin= {stdin}, *yyout = {stdout};
extern int yylineno;
struct yysvf {
    struct yyork *yystoff;
    struct yysvf *yyother;    
    int *yystops;
};
stuct yysvf *yyestate;
extern struct  yysvf yysvec[], *yybgin;

#include <string.h>
#include "util.h"
#include "symbol.h"
#include "absyn.h"
#include "y.tab.h"
#include "errormsg.h"

static int comLevel=0;

#define STRINGMAX 1024
char stringbuild[STRINGMAX+1];
int stringindex=0;

static void append(char c)
{
    if (stringindex < STRINGMAX) {
        stringbuild[stringindex++] = c;
    } else {
        EM_error(EM_tokPos, "string too long.");
        stringindex=0;
    }
}

static string getstring(void) {
    stringbuild[stringindex]=0;
    stringindex=0;
    return String(stringbuild);
}

int charPos=1;

int yywrap(void) {
    if (comLevel) {
        EM_error(EM_tokPos, "unclosed comment");
        charPos=1;
        return 1;
    }
}

void adjust() {
    EM_tokPos = charPos;
    charPos += yyleng;
}

#define A 2
#define S 4
#define F 6
#define YYNEWLINE 10
int yylex() {
    int nstr;
    extern int yyprevious;
    while ((nstr = yylook()) >=0) {
       yyfussy: switch (nstr)
       {
        case 0:
            if(yywrap()) return 0;
            break;
        case 1:{
            adjust();
            continue;
        }
        break;
        case 2:
        {
            adjust();
            EM_newline();
            continue;
        }
        break;
        case 3:
        {
            adjust();
            return COMMA;
        }
        break;
       default:
        break;
       } 
    }
    return 0;
}
