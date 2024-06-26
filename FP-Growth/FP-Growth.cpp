// #include <bits/stdc++.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <cstring>
#include <sstream>
#include <algorithm>
#include <set>
#include <iomanip>
using namespace std;
// int totalNum;
int itemSupport[1005];
int headerPos[1005];
double supportNum;
int numOfItemSet = 0;
// set<vector<int> > freqItemSet;
vector<pair<vector<int>, double> > freqItemSet;
struct TreeNode{
public:
    int count;
    int id;
    vector<TreeNode*> children;
    TreeNode* parentNode;
    TreeNode(int id){
        count = 0;
        this->id = id;
        parentNode=nullptr;
    }
};

struct headNode{
    pair<int, int> itemToFreq;
    vector<TreeNode*> next;
};

bool compare(const int a, const int b)
{
    if(itemSupport[a] != itemSupport[b])    return itemSupport[a] > itemSupport[b];
    else return a < b;
}

void parseLine(string line, vector<int> &data){
    string temp = "";
    for(int i=0;i<line.length();i++){
        if(line[i] ==','){
            data.push_back(atoi(temp.c_str()));
            itemSupport[atoi(temp.c_str())]++;
            temp = "";
        }
        else temp += line[i];
    }
    data.push_back(atoi(temp.c_str()));
    itemSupport[atoi(temp.c_str())]++;
}
void readFile( vector<vector<int> >& transaction, char* argv[], vector<headNode*>& headerTable)
{
    float c = stod(argv[1]);
    int zero = 0;
    ifstream inputFile(argv[2]);
    // if (!inputFile) {
    //     cerr << "Unable to open input file: " << argv[2] << '\n';
    //     return 1;
    // }

    string line;
    while (getline(inputFile, line)) {
        vector<int> data;
        // parseLine(line, data);
        stringstream line_stream(line);
        string num;
        while (getline(line_stream, num, ',')) {
            // outputFile << num << " , ";
            data.push_back(atoi(num.c_str()));
            itemSupport[atoi(num.c_str())]++;
        }
        transaction.push_back(data);

    }
    inputFile.close();

    // cout<<"ratio = "<<c<<endl;
    // cout<<"transaction size = "<<transaction.size()<<endl;
    int totalNum = transaction.size();
    // cout<<"total = "<<totalNum<<endl;
    supportNum = totalNum * c;
    cout<<"supportNum: "<<supportNum<<endl;
    for(int i=0; i<1005; i++){
        if(itemSupport[i] >= supportNum){
            headNode* node = new headNode();
            node->itemToFreq = make_pair(i, itemSupport[i]);
            headerTable.push_back(node);
        }
    }

    sort(headerTable.begin(), headerTable.end(), [](const headNode* a, const headNode* b){
        return a->itemToFreq.second > b->itemToFreq.second;
    });
    // cout<<"----------------headerTable----------------------------------"<<endl;
    for(int i=0; i<headerTable.size(); i++){
        // cout<<headerTable[i]->itemToFreq.first<<" "<<headerTable[i]->itemToFreq.second<<" "<<endl;
        headerPos[headerTable[i]->itemToFreq.first] = i;//headerPos[item]=index
    }
    //  cout<<"---------------------------------------------------------------"<<endl;

}

void buildTree(vector<int>::iterator it, vector<int>::iterator end, vector<headNode*>& headerTable, TreeNode* root, int c)
{
    if(it == end)   return;
    bool flag = false;
    int len = root->children.size();
    for(int i=0; i<len; i++){
        TreeNode* child = root->children[i];
        if(child->id == *it){
            child->count += c;
            buildTree(it+1, end, headerTable, child, c);
            flag = true;
            break;
        }
    }
    //tree中沒有相同節點
    if(!flag){
        while(it != end){
            TreeNode* node = new TreeNode(*it);
            node->parentNode = root;
            node->count = c;
            headerTable[headerPos[(*it)]]->next.push_back(node);
            root->children.push_back(node);
            root = node;
            it++;
        }
    }
}

void printTree(TreeNode* root, int level)
{
    // cout<<"printTree"<<endl;
    int len = root->children.size();
    for(int i=0; i<len; i++){
        cout<<endl<<"--------------------------level "<<level<<"----------------------------------"<<endl;
        cout<<root->children[i]->id<<" "<<root->children[i]->count<<" "<<endl;
        int l = level + 1;
        printTree(root->children[i], l);
    }
}

void printHeaderTable(vector<headNode*>& headerTable)
{
    for(int i=0; i<headerTable.size(); i++){
        cout<<"headerTable["<<i<<"]->next.size(): "<<headerTable[i]->next.size()<<endl;
        for(int j=0; j<headerTable[i]->next.size(); j++){
            cout<<"----------------------headerTable"<<i<<"--------------------------"<<endl;
            cout<<headerTable[i]->next[j]->id<<" "<<headerTable[i]->next[j]->count<<" "<<endl;
        }
    }
}

vector<int> getPattern(TreeNode* node, vector<int> vec){
    if(node->id == -1)  return vec;
    vec.push_back(node->id);
    // cout<<node->id<<" ";
    return getPattern(node->parentNode, vec);
}

void FP_Growth(vector<headNode*>& headerTable, vector<int> freqSet){
    int len = headerTable.size();
    for(int i=len-1; i>=0; i--){//從headerTable最下面開始找(最下面frequency最低)
        // vector<pair<vector<int>, int> > condPatternBase; //(conditional pattern, frequncy)
        vector<pair<vector<int>, int> > condPatternBase; //(conditional pattern, frequncy)
        vector<headNode*> condHeaderTable;
        vector<int> data;
        memset(itemSupport, 0, sizeof(itemSupport));
        memset(headerPos, -1, sizeof(headerPos));
        for(int j=0; j<headerTable[i]->next.size(); j++){//找所有和該header table的item一樣的item
            // cout<<"headerTable["<<i<<"]->next["<<j<<"]->id: "<<headerTable[i]->next[j]->id<<endl;
            vector<int> tmp;
            vector<int> condPattern = getPattern(headerTable[i]->next[j]->parentNode, tmp);//record frequency pattern candidate
            // cout<<endl;
            sort(condPattern.begin(), condPattern.end());
            condPatternBase.push_back(make_pair(condPattern, headerTable[i]->next[j]->count));
            for(auto pattern: condPattern){
                itemSupport[pattern] += headerTable[i]->next[j]->count;//recompute each item support
                data.push_back(pattern);
            }
        }
        sort(data.begin(), data.end());//delete duplicate item after sorting
        data.erase(unique(data.begin(), data.end()), data.end());
        sort(data.begin(), data.end(), compare);
        int ind = 0;
        for(auto it = data.begin(); it!= data.end(); it++){
            if(itemSupport[*it] < supportNum){
                data.erase(it, data.end());//把support太小的都丟掉，現在data剩下大於supportNum的item
                break;
            }
            headNode* node = new headNode();
            node->itemToFreq = make_pair(data[ind], itemSupport[data[ind]]);
            condHeaderTable.push_back(node);
            headerPos[data[ind]] = ind;
            ++ind;
        }
        TreeNode* condRoot = new TreeNode(-1);
        for(auto &row:condPatternBase){
            stable_sort(row.first.begin(), row.first.end(), compare);
            for(auto it=row.first.begin(); it!=row.first.end(); it++){
                // cout<<"itemSupport["<<*it<<"]: "<<itemSupport[*it] <<" "<<endl;
                if(itemSupport[*it] < supportNum){
                    row.first.erase(it, row.first.end());
                    break;
                }
            }
            buildTree(row.first.begin(), row.first.end(), condHeaderTable,  condRoot, row.second);
        }
        vector<int> newFreqSet = freqSet;
        newFreqSet.push_back(headerTable[i]->itemToFreq.first);
        sort(newFreqSet.begin(), newFreqSet.end());
        if(freqSet.empty()){
            freqItemSet.push_back(make_pair(newFreqSet, headerTable[i]->itemToFreq.second));
        }

        // freqItemSet.insert(newFreqSet);
        for(auto item:data){
            vector<int> tmp = newFreqSet;
            tmp.push_back(item);
            sort(tmp.begin(), tmp.end());
            freqItemSet.push_back(make_pair(tmp, itemSupport[item]));

            // freqItemSet.insert(tmp);
        }
        if(condHeaderTable.size())     FP_Growth(condHeaderTable, newFreqSet);
    }
}

void printFreqItemSet(vector<vector<int> > transaction, string output, int totalNum){
    cout<<"numOfFreqsItemSet: "<<freqItemSet.size()<<endl;
    // sort(freqItemSet.begin(), freqItemSet.end(), [](const pair<vector<int>, int> & a, const pair<vector<int>, int> & b){
    //     if (a.first.size() != b.first.size()) {
    //         return a.first.size() < b.first.size();
    //     }
    //     int i = 0;
    //     for (i = 0; i < a.first.size(); ++i) {
    //         if (a.first[i] != b.first[i])
    //             break;
    //     }
    //     return a.first[i] < b.first[i];
    // });

    ofstream outputFile(output);
    // for(auto freqItem=freqItemSet.begin(); freqItem!=freqItemSet.end(); freqItem++){
    for(int i=0; i<freqItemSet.size(); i++){
        // double total = 0;
        // int transaction_idx = 1;
        // for(auto item: transaction){
        //     sort(item.begin(), item.end());
        //     int tmp = 0;
        //     for(int i = 0; i<item.size(); i++){
        //         // if(item[i] == (*freqItem)[tmp]) ++tmp;
        //         // if(tmp == (*freqItem).size()){
        //         //     ++total;
        //         //     break;
        //         // }
        //         if(item[i] == freqItem[tmp]) ++tmp;
        //         if(tmp == freqItem.size()){
        //             ++total;
        //             break;
        //         }
        //     }
        //     // cout<<endl;
        //     transaction_idx++;
        // }
        // cout<<endl;
        // double sup = total/totalNum;
        // for(int i = 0; i < (*freqItem).size(); i++){
        //     if(i == (*freqItem).size()-1)  outputFile<<(*freqItem)[i];
        //     else    outputFile<<(*freqItem)[i]<<",";
        // }
        for(int j = 0; j < freqItemSet[i].first.size(); j++){
            if(j ==  freqItemSet[i].first.size()-1)  outputFile<<freqItemSet[i].first[j];
            else    outputFile<<freqItemSet[i].first[j]<<",";
        }

        double sup = freqItemSet[i].second/totalNum;
        outputFile<<":"<<fixed<<setprecision(4)<<sup;
        outputFile<<endl;
        // system("pause");
    }
    outputFile.close();
}

int main(int argc, char* argv[])
{
    if (argc != 4) cerr << "Usage: " << argv[0] << " <scale> <inputfile> <outputfile>" << endl;
    memset(itemSupport, 0, sizeof(itemSupport));
    memset(headerPos, -1, sizeof(headerPos));
    vector<vector<int> > transaction;
    vector<headNode*> headerTable;
    readFile(transaction, argv, headerTable);
    TreeNode* root = new TreeNode(-1);
    int totalNum = transaction.size();
    for(auto &row: transaction){
        stable_sort(row.begin(), row.end(), compare);
        for(auto it=row.begin(); it!=row.end(); it++){
            if(itemSupport[*it] < supportNum){
                row.erase(it, row.end());
                break;
            }
        }
        buildTree(row.begin(),row.end(), headerTable,  root, 1);
    }
    // printTree(root, 1);
    // printHeaderTable(headerTable);
    vector<int> freqSet;
    FP_Growth(headerTable, freqSet);
    cout<<"totalNum = "<<totalNum<<endl;
    printFreqItemSet(transaction, argv[3], totalNum);
    return 0;
}