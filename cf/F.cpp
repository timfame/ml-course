#include <iostream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <iomanip>

using namespace std;

int k;
long double lc[10];
long double alpha;
int n, m;
vector <int> c, l;
vector <unordered_map <string, int>> words;
vector <unordered_map <string, long double>> prob;
unordered_set <string> all_words;
int class_cnt[10];
long double zn[10];
long double results[10];
unordered_set <string> current_test;

int main() {
    cin >> k;
    for (int i = 0; i < k; ++i) {
        cin >> lc[i];
    }
    cin >> alpha;
    cin >> n;
    c.resize(n); l.resize(n);
    words.resize(k); prob.resize(k);
    for (int i = 0; i < n; ++i) {
        cin >> c[i] >> l[i];
        current_test.clear();
        --c[i];
        for (int j = 0; j < l[i]; ++j) {
            string word;
            cin >> word;
            current_test.insert(word);
            all_words.insert(word);
        }
        for (string w : current_test) {
            words[c[i]][w]++;
        }
        class_cnt[c[i]]++;
    }
    for (int cl = 0; cl < k; ++cl) {
        zn[cl] = (long double)(class_cnt[cl]) + alpha * 2.0;
        for (pair<string, int> w : words[cl]) {
            long double ch = (long double)(w.second) + alpha;
            prob[cl][w.first] = ch / zn[cl];
        }
    }

    // im try to figure out
    // am i correct
    //
    cin >> m;
    for (int test = 0; test < m; ++test) {
        current_test.clear();
        int test_cnt;
        cin >> test_cnt;
        for (int i = 0; i < test_cnt; ++i) {
            string word;
            cin >> word;
            current_test.insert(word);
        }
        long double sum = 0;
        for (int cl = 0; cl < k; ++cl) {
            long double result = (long double)(class_cnt[cl]) / (long double)(n);
            for (string w : all_words) {
                long double current_prob = alpha / zn[cl];
                if (prob[cl].find(w) != prob[cl].end()) {
                    current_prob = prob[cl][w];
                }
                if (current_test.find(w) == current_test.end()) {
                    current_prob = 1 - current_prob;
                }
                result *= current_prob;
            }
            result *= lc[cl];
            results[cl] = result;
            sum += result;
        }
        for (int cl = 0; cl < k; ++cl) {
            results[cl] /= sum;
        }
        for (int cl = 0; cl < k; ++cl) {
            cout << fixed << setprecision(10) << results[cl] << " ";
        }
        if (test < m - 1) {
            cout << "\n";
        }
    }
}