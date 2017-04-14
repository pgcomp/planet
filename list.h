#ifndef LIST_H
#define LIST_H

#include <cstdlib> // malloc, free
#include <cassert>

template<class T>
struct List
{
    T *data;
    int num;
    int cap;
};

template<class T>
void ListReserve(List<T> &list, int n)
{
    assert(n > 0);
    if (list.cap < n)
    {
        T *new_data = (T*)malloc(n * sizeof(T));
        assert(new_data != nullptr);
        assert(((uintptr_t)new_data & (alignof(T) - 1)) == 0);

        if (list.data)
        {
            if (list.num > 0)
            {
                memcpy(new_data, list.data, list.num * sizeof(T));
            }
            free(list.data);
        }

        list.data = new_data;
        list.cap = n;
    }
}

template<class T>
void ListResize(List<T> &list, int n)
{
    assert(n >= 0);
    if (list.cap < n)
    {
        ListReserve(list, n);
    }
    list.num = n;
}

template<class T>
void ListAdd(List<T> &list, T value)
{
    if (list.num == list.cap)
    {
        int n = Max(list.cap * 2, 96);
        ListReserve(list, n);
    }

    list.data[list.num++] = value;
}

template<class T>
void ListFree(List<T> &list)
{
    if (list.data)
    {
        free(list.data),
        list.data = nullptr;
        list.num = 0;
        list.cap = 0;
    }
}

#endif // LIST_H
