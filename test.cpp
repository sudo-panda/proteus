#include <iostream>

char *str;
int a[100] = { 0 };
int b[100] = { 0 };

int foo(int, int);

__attribute__((annotate("jit",1)))
void bar(int a)
{
    printf("a %d\n", a);
}

__attribute__((annotate("jit",1,2)))
int foo(int v, int v2)
{
    if (v == 1)
        printf("1 v %d\n", v);
    else
        printf("not 1 v %d\n", v);
    bar(10);
    return v+1;
}

int main()
{
    int v = -1;
    str = strdup("hello1");
    for(int i=0;i<100000; ++i) {
        v = -1;
        foo(v,0);
        printf("v %d\n", v);
    }
    v = 40;
    str = strdup("hello2");
    v = foo(v,0);
    printf("v %d\n", v);

    bar(20);

    printf("done...\n");

    return 0;
}
