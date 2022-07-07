//
// Created by Dylan on 6/2/2022.
//

#ifndef CRUSADER_THREADCONTROLLER_CUH
#define CRUSADER_THREADCONTROLLER_CUH

#include <thread>
#include <vector>


using namespace std;
namespace seutil{

    //To use this, the first Xs of your function need to be the tid
    template<const int x, const int y, typename ...Args, typename ...Args0>
    void _alloc(void(*func)(Args...), Args0... args){
        vector<thread> threads;
        for(int i = 0; i < x; i++){
            for (int j = 0; j < y; j++){
                threads.push_back(thread(func, i, j, args...));
            }
        }

        for(auto& t : threads){
            t.join();
        }
    }

    template<const int x, typename ...Args, typename ...Args0>
    void _alloc(void(*func)(Args...), Args0... args){
        vector<thread> threads;
        threads.reserve(x);
        for(int i = 0; i < x; i++){
            threads.push_back(thread(func, i, x, args...));
        }

        for(auto& t : threads){
            t.join();
        }
    }
}


#endif //CRUSADER_THREADCONTROLLER_CUH
