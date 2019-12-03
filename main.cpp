#include <iostream>
#include <Python.h>

using namespace std;

int main()
{   

    char filename[] = "/home/velab/VsCode++/TestEmbedPython/lib/main.py";
    FILE *fp;

    Py_Initialize();

    fp = _Py_fopen(filename, "r");
    PyRun_SimpleFile(fp, filename);

    Py_Finalize();
    return 0;
    cout << "test" << endl;

    return 0;
}