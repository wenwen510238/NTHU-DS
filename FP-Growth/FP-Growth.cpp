#include <bits/stdc++.h>
using namespace std;

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

// float minSupport=0;
int totalNum=0;
int itemSupport[1000];
int headerPos[1000];
vector<pair<vector<int>, int> > outputData;
double supportNum;

bool compare(const int a, const int b){
    return itemSupport[a] > itemSupport[b];
}
void readFile( vector<vector<int> >& origin_data, char* argv[], vector<headNode*>& headerTable){
    // 檢查命令列參數數量
    float c;
    // 解析浮點數參數
    istringstream scale_str(argv[1]);
    scale_str>>c;
    // double scale;
    // if (!(scale_str >> scale)) {
    //     cout << "min support: " << argv[1] << endl;
    //     return 1;
    // }

    // 打開輸入文件
    ifstream inputFile(argv[2]);
    // if (!inputFile) {
    //     cerr << "Unable to open input file: " << argv[2] << '\n';
    //     return 1;
    // }

    // 打開輸出文件
    // ofstream outputFile(argv[3]);
    // if (!outputFile) {
    //     cerr << "Unable to open output file: " << argv[3] << '\n';
    //     return 1;
    // }

    // 讀取輸入文件，每次處理一行
    string line;
    while (getline(inputFile, line)) {
        vector<int> data;
        istringstream line_stream(line);
        string num;
        while (getline(line_stream, num, ',')) {
            // outputFile << num << " , ";
            data.push_back(atoi(num.c_str()));
            itemSupport[atoi(num.c_str())]++;
        }
        totalNum++;
        origin_data.push_back(data);
        // outputFile << '\n';
    }
    inputFile.close();

    supportNum = totalNum*c;
    cout<<"supportNum: "<<supportNum<<endl;
    vector<pair<int, int> > itemToFreq;
    for(int i=0; i<1000; i++){
        if(itemSupport[i] >= supportNum){
            headNode* node = new headNode();
            node->itemToFreq = make_pair(i, itemSupport[i]);
            headerTable.push_back(node);
        }
    }

    sort(headerTable.begin(), headerTable.end(), [](const headNode* a, const headNode* b){
        return a->itemToFreq.second > b->itemToFreq.second;
    });
    cout<<"----------------headerTable----------------------------------"<<endl;
    for(int i=0; i<headerTable.size(); i++){
        cout<<headerTable[i]->itemToFreq.first<<" "<<headerTable[i]->itemToFreq.second<<" "<<endl;
        headerPos[headerTable[i]->itemToFreq.first] = i;//headerPos[item]=index
    }
     // cout<<"---------------------------------------------------------------"<<endl;

}
void buildTree(vector<int>::iterator it, vector<int>::iterator end, vector<headNode*>& headerTable, TreeNode* root){
    if(it == end)   return;
    bool flag = false;
    int len = root->children.size();
    for(int i=0; i<len; i++){
        TreeNode* child = root->children[i];
        if(child->id == *it){
            child->count++;
            flag = true;
            buildTree(it+1, end, headerTable, child);
            break;
        }
    }
    //tree中沒有相同節點
    if(!flag){
        while(it!= end){
            TreeNode* node = new TreeNode(*it);
            node->parentNode = root;
            node->count++;
            headerTable[headerPos[(*it)]]->next.push_back(node);
            root->children.push_back(node);
            root = node;
            it++;
        }
    }
}
void printTree(TreeNode* root, int level){
    int len = root->children.size();
    for(int i=0; i<len; i++){
        cout<<endl<<"--------------------------level "<<level<<"----------------------------------"<<endl;
        cout<<root->children[i]->id<<" "<<root->children[i]->count<<" "<<endl;
        int l = level + 1;
        printTree(root->children[i], l);
    }
}
void printHeaderTable(vector<headNode*>& headerTable){
    for(int i=0; i<headerTable.size(); i++){
        for(int j=0; j<headerTable[i]->next.size(); j++){
            cout<<"----------------------headerTable"<<i<<"--------------------------"<<endl;
            cout<<headerTable[i]->next[j]->id<<" "<<headerTable[i]->next[j]->count<<" "<<endl;
        }
    }
}
vector<int> getPattern(TreeNode* node, vector<int> vec){
    if(node->id == -1)  return vec;
    vec.push_back(node->id);
    return getPattern(node->parentNode, vec);
}
void fPGrowth(vector<headNode*>& headerTable){
    int len = headerTable.size();
    for(int i=len-1; i>=0; i--){//從headerTable最下面開始找(最下面frequency最低)
        vector<pair<vector<int>, int> > condPatternBase; //(conditional pattern, frequncy)
        vector<int> data;
        memset(itemSupport, 0, 1000 * sizeof(itemSupport[0]));
        for(int j=0; j<headerTable[i]->next.size(); j++){//找所有和該header table的item一樣的item
            vector<int> tmp;
            vector<int> condPattern = getPattern(headerTable[i]->next[j], tmp);//record frequency pattern candidate
            condPatternBase.push_back(make_pair(condPattern, headerTable[i]->next[j]->count));
            for(auto pattern: condPattern){
                itemSupport[pattern] += headerTable[i]->next[j]->count;//重新計匴每個item的support
                data.push_back(pattern);
            }
        }
        sort(data.begin(), data.end());//排序後才再透過unique刪除重複的
        data.erase(unique(data.begin(), data.end()), data.end());
        sort(data.begin(), data.end(), compare);
        for(auto it = data.begin(); it!= data.end(); it++){
            if(itemSupport[*it] < supportNum){
                data.erase(it, data.end());//把support太小的都丟掉，現在data剩下大於supportNum的item
                break;
            }
        }


    }
}
int main(int argc, char* argv[]) {
    if (argc != 4) cerr << "Usage: " << argv[0] << " <scale> <inputfile> <outputfile>" << endl;
    memset(itemSupport, 0, 1000 * sizeof(itemSupport[0]));
    memset(headerPos, 0, 1000 * sizeof(headerPos[0]));
    vector<vector<int> > origin_data;
    vector<headNode*> headerTable;
    readFile(origin_data, argv, headerTable);
    TreeNode* root = new TreeNode(-1);

    //----------------remove number which less than min supportNum---------------------------------//
    for(auto &row:origin_data){
        stable_sort(row.begin(), row.end(), compare);
        for(auto it=row.begin(); it!=row.end(); it++){
            if(itemSupport[*it] < supportNum){
                row.erase(it, row.end());
                break;
            }
        }
        buildTree(row.begin(),row.end(), headerTable,  root);
    }
    // printTree(root, 1);
    // printHeaderTable(headerTable);
    // fPGrowth(headerTable);

    //---------------------------------------------------------------------------------------------//

    // cout<<"-----------------after erasing origin_data----------------------"<<endl;
    // for(auto &row:origin_data){
    //     for(auto it=row.begin(); it!=row.end(); it++){
    //         cout<<*it<<" ";
    //     }
    //     cout<<endl;
    // }
    // cout<<"----------------------------------------------------------------"<<endl;
    // buildTree(origin_data, headerTable, root);


    // outputFile.close();
    return 0;
}