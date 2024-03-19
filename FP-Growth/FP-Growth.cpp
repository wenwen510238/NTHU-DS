#include <iostream>
#include <string>
#include <vector>
#include <fstream>  
#include <sstream>
using namespace std;

class TrieNode{
public:
    bool count;
    TrieNode* childNode[1000];
    TrieNode* parentNode[1000];
    TrieNode(){
        count  = 0;
        for(int i=0; i<1000; i++) parentNode[i]=NULL;
    }
};
int main(int argc, char* argv[]) {
    // 檢查命令列參數數量
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <scale> <inputfile> <outputfile>" << endl;
        return 1;
    }

    // 解析浮點數參數
    istringstream scale_str(argv[1]);
    double scale;
    if (!(scale_str >> scale)) {
        cout << "min support: " << argv[1] << endl;
        return 1;
    }

    // 打開輸入文件
    ifstream input(argv[2]);
    if (!input) {
        cerr << "Unable to open input file: " << argv[2] << '\n';
        return 1;
    }

    // 打開輸出文件
    ofstream output(argv[3]);
    if (!output) {
        cerr << "Unable to open output file: " << argv[3] << '\n';
        return 1;
    }

    // 讀取輸入文件，每次處理一行
    string line;
    while (getline(input, line)) {
        istringstream line_stream(line);
        double num;
        while (line_stream >> num) {
            output << num << ' ';
        }
        output << '\n';
    }

    // 關閉文件
    input.close();
    output.close();

    return 0;
}