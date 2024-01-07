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
        case 4:
        {
            adjust();
            return LBRACE;
        }
        break;
        case 5:
        {
            adjust();
            return RBRACE;
        }
        break;
        case 6:
        {
            adjust(); 
            return LBRACK;
        }
        break;
        case 7:
        {
            adjust(); 
            return RBRACK;
        }
        break;
        case 8:
            {adjust(); return COLON;}
        break;
        case 9:
        {adjust(); return SEMICOLON;}
        break;
        case 10:
        {adjust(); return LPAREN;}
        break;
        case 11:
        {adjust(); return RPAREN;}
        break;
        case 12:
        {adjust(); return DOT;}
        break;
        case 13:
        {adjust(); return PLUS;}
        break;
        case 14:
        {adjust(); return MINUS;}
        break;
        case 15:
        {adjust(); return TIMES;}
        break;
        case 16:
        {adjust(); return DIVIDE;}
        break;
        case 17:
        {adjust(); return AND;}
        break;
        case 18:
        {adjust(); return OR;}
        break;
        case 19:
        {adjust(); return ASSIGN;}
        break;
        case 20:
        {adjust(); return EQ;}
        break;
        case 21:
        {adjust(); return NEQ;}
        break;
        case 22:
        {adjust(); return GT;}
        break;
        case 23:
        {adjust(); return LT;}
        break;
        case 24:
        {adjust(); return GE;}
        break;
        case 25:
        {adjust(); return LE;}
        break;
        case 26:
            {adjust(); return FOR;}
        break;
        case 27:
        {adjust(); return WHILE;}
        break;
        case 28:
        {adjust(); return BREAK;}
        break;
        case 29:
            {adjust(); return LET;}
        break;
        case 30:
            {adjust(); return IN;}
        break;
        case 31:
            {adjust(); return NIL;}
        break;
        case 32:
            {adjust(); return TO;}
        break;
        case 33:
            {adjust(); return END;}
        break;
        case 34:
        {adjust(); return FUNCTION;}
        break;
        case 35:
            {adjust(); return VAR;}
        break;
        case 36:
            {adjust(); return TYPE;}
        break;
        case 37:
            {adjust(); return ARRAY;}
        break;
        case 38:
            {adjust(); return IF;}
        break;
        case 39:
            {adjust(); return THEN;}
        break;
        case 40:
            {adjust(); return ELSE;}
        break;
        case 41:
            {adjust(); return DO;}
        break;
        case 42:
            {adjust(); return OF;}
        break;
        case 43:
        {adjust(); yylval.sval = String((char*)yytext); return ID;}
        break;
        case 44:
        {adjust(); yylval.ival=atoi(yytext); return INT;}
        break;
        case 45:
        {adjust(); BEGIN S; continue;}
        break;
        case 46:
        {adjust(); BEGIN A; comLevel = 1; continue;}
        break;
        case 47:
        {adjust(); EM_error(EM_tokPos,"unmatched close comment");
                    continue;}
        break;
        case 48:
        {adjust(); EM_error(EM_tokPos,"non-Ascii character");
                    continue;}
        break;
        case 49:
        {adjust(); EM_error(EM_tokPos,"illegal token");
                    continue;}
        break;
        case 50:
            {adjust(); comLevel++; continue;}
        break;
        case 51:
            {adjust(); EM_newline(); continue;}
        break;
        case 52:
            {adjust(); comLevel--; 
                    if (comLevel==0) {BEGIN INITIAL;}
                            continue;}
        break;
        case 53:
            {adjust(); continue;}
        break;
        case 54:
            {adjust(); BEGIN INITIAL; 
                        yylval.sval=getstring();
                        return STRING;}
        break;
        case 55:
            {adjust(); EM_error (EM_tokPos,"unclosed string");
                        EM_newline();
                    BEGIN INITIAL; 
                        yylval.sval=getstring();
                    return STRING;}
        break;
        case 56:
                {adjust(); EM_newline(); BEGIN F; continue;}
        break;
        case 57:
            {adjust(); BEGIN F; continue;}
        break;
        case 58:
            {adjust(); append(*yytext); continue;}
        break;
        case 59:
            {adjust(); EM_newline(); continue;}
        break;
        case 60:
            {adjust(); continue;}
        break;
        case 61:
            {adjust(); BEGIN S; continue;}
        break;
        case 62:
            {adjust(); EM_error(EM_tokPos, "unclosed string"); 
                    BEGIN INITIAL; 
                    yylval.sval=getstring();
                    return STRING;}
        break;
        case 63:
            {adjust(); append('\t'); continue;}
        break;
        case 64:
            {adjust(); append('\n'); continue;}
        break;
        case 65:
            {adjust(); append('\\'); continue;}
        break;
        case 66:
        {adjust(); append(yytext[1]); continue;}
        break;
        case 67:
        {adjust(); append(yytext[2]-'@');
                        continue;}
        break;
        case 68:
        {int x = yytext[1]*100 + yytext[2]*10 + yytext[3] - 
                                ('0' * 111);
                    adjust();
                            if (x>255)
                                EM_error(EM_tokPos, "illegal ascii escape");
                        else append(x);
                        continue;
                    }
        break;
        case 69:
            {adjust(); EM_error(EM_tokPos, "illegal string escape"); 
                    continue;}
        break;
        case -1:
        break;
        default:
        fprintf(yyout,"bad switch yylook %d",nstr);
       } 
    }
    return 0;
}
