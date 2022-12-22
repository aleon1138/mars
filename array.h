#include <cassert>
#include <cstdlib>

template <typename T> void fill(T *p, T val, int len)
{
    for (int i = 0; i < len; ++i) {
        p[i] = val;
    }
}


template <typename T>
class array_t {
    T  *_ptr;
    int _len;
public:
    array_t(int len)
    {
        assert(len >= 0);
        _ptr = (T*)aligned_alloc(64, len*sizeof(T));
        _len = len;
    }

    array_t(T val, int len): array_t(len)
    {
        fill(_ptr, val, len);
    }

    /*
     *  Move constructor.
     */
    array_t(array_t&& x);

    /*
     *  Destructor.
     */
    ~array_t()
    {
        free(_ptr);
    }

    T& operator[](int i)
    {
        assert(i >= 0 && i < _len);
        return _ptr[i];
    }

    const T& operator[](int i) const
    {
        assert(i >= 0 && i < _len);
        return _ptr[i];
    }

    operator T *()
    {
        return _ptr;
    }

    int len() const
    {
        return _len;
    }

    void trim(int len)
    {
        assert(len >= 0 && len <=_len);
        _len = len;
    }

private:
    array_t(const array_t& x);
    array_t& operator=(const array_t& x);
};

