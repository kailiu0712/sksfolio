#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>


typedef struct {
    double value;
    Py_ssize_t index;
} HeapItem;


typedef struct {
    double point;
    double alpha;
    double beta;
    unsigned char is_top;
} Breakpoint;


enum NativeStatus {
    NATIVE_OK = 0,
    NATIVE_MEMORY_ERROR = 1,
    NATIVE_NONFINITE_ARGUMENT = 2,
    NATIVE_UNSAFE_SCALE = 3
};


static double
singleton_subtraction(double value, double gamma)
{
    if (value <= 1.0 + gamma) {
        return (gamma / (1.0 + gamma)) * value;
    }
    return value - 1.0;
}


static int
heap_less(const HeapItem *left, const HeapItem *right)
{
    if (left->value < right->value) {
        return 1;
    }
    if (left->value > right->value) {
        return 0;
    }
    return left->index > right->index;
}


static void
heap_sift_down(HeapItem *heap, Py_ssize_t size, Py_ssize_t root)
{
    for (;;) {
        Py_ssize_t child = 2 * root + 1;
        Py_ssize_t smallest = root;
        if (child < size && heap_less(&heap[child], &heap[smallest])) {
            smallest = child;
        }
        if (
            child + 1 < size
            && heap_less(&heap[child + 1], &heap[smallest])
        ) {
            smallest = child + 1;
        }
        if (smallest == root) {
            return;
        }
        HeapItem temporary = heap[root];
        heap[root] = heap[smallest];
        heap[smallest] = temporary;
        root = smallest;
    }
}


static int
heap_item_is_better(const HeapItem *candidate, const HeapItem *root)
{
    if (candidate->value > root->value) {
        return 1;
    }
    if (candidate->value < root->value) {
        return 0;
    }
    return candidate->index < root->index;
}


static double
median_of_three(double first, double middle, double last)
{
    if (first > middle) {
        double temporary = first;
        first = middle;
        middle = temporary;
    }
    if (middle > last) {
        double temporary = middle;
        middle = last;
        last = temporary;
    }
    if (first > middle) {
        middle = first;
    }
    return middle;
}


static double
mixed_pool_boundary(
    const double *top,
    Py_ssize_t top_size,
    const double *tail,
    Py_ssize_t tail_size,
    double gamma,
    double *top_breakpoints,
    Breakpoint *breakpoints
)
{
    double smallest_top = DBL_MAX;
    for (Py_ssize_t index = 0; index < top_size; ++index) {
        double breakpoint = singleton_subtraction(top[index], gamma);
        top_breakpoints[index] = breakpoint;
        if (breakpoint < smallest_top) {
            smallest_top = breakpoint;
        }
    }
    if (tail_size == 0) {
        return smallest_top;
    }

    double largest_tail = -DBL_MAX;
    for (Py_ssize_t index = 0; index < tail_size; ++index) {
        if (tail[index] > largest_tail) {
            largest_tail = tail[index];
        }
    }
    if (largest_tail <= smallest_top) {
        return smallest_top;
    }

    double switch_point = gamma;
    double value_at_switch = 0.0;
    for (Py_ssize_t index = 0; index < top_size; ++index) {
        double value = (
            (gamma + 1.0) * switch_point - gamma * top[index]
        );
        if (value > 0.0) {
            value_at_switch += value;
        }
    }
    for (Py_ssize_t index = 0; index < tail_size; ++index) {
        if (tail[index] > switch_point) {
            value_at_switch += gamma * (switch_point - tail[index]);
        }
    }

    double certified_alpha = 0.0;
    double certified_beta = 0.0;
    Py_ssize_t count = 0;
    if (value_at_switch > 0.0) {
        for (Py_ssize_t index = 0; index < top_size; ++index) {
            if (top_breakpoints[index] < switch_point) {
                breakpoints[count].point = top_breakpoints[index];
                breakpoints[count].alpha = gamma + 1.0;
                breakpoints[count].beta = gamma * top[index];
                breakpoints[count].is_top = 1;
                ++count;
            }
        }
        for (Py_ssize_t index = 0; index < tail_size; ++index) {
            if (tail[index] >= switch_point) {
                certified_alpha += gamma;
                certified_beta += gamma * tail[index];
            }
            else {
                breakpoints[count].point = tail[index];
                breakpoints[count].alpha = gamma;
                breakpoints[count].beta = gamma * tail[index];
                breakpoints[count].is_top = 0;
                ++count;
            }
        }
    }
    else if (value_at_switch < 0.0) {
        for (Py_ssize_t index = 0; index < top_size; ++index) {
            if (top_breakpoints[index] <= switch_point) {
                certified_alpha += 1.0;
                certified_beta += top[index] - 1.0;
            }
            else {
                breakpoints[count].point = top_breakpoints[index];
                breakpoints[count].alpha = 1.0;
                breakpoints[count].beta = top[index] - 1.0;
                breakpoints[count].is_top = 1;
                ++count;
            }
        }
        for (Py_ssize_t index = 0; index < tail_size; ++index) {
            if (tail[index] > switch_point) {
                breakpoints[count].point = tail[index];
                breakpoints[count].alpha = 1.0;
                breakpoints[count].beta = tail[index];
                breakpoints[count].is_top = 0;
                ++count;
            }
        }
    }
    else {
        return switch_point;
    }

    while (count > 0) {
        double pivot = median_of_three(
            breakpoints[0].point,
            breakpoints[count / 2].point,
            breakpoints[count - 1].point
        );
        double alpha = certified_alpha;
        double beta = certified_beta;
        for (Py_ssize_t index = 0; index < count; ++index) {
            const Breakpoint *item = &breakpoints[index];
            if (
                (item->is_top && item->point < pivot)
                || (!item->is_top && item->point > pivot)
            ) {
                alpha += item->alpha;
                beta += item->beta;
            }
        }
        double value = alpha * pivot - beta;
        double scale = 1.0 + fabs(alpha * pivot) + fabs(beta);
        if (fabs(value) <= 8.0 * DBL_EPSILON * scale) {
            return pivot;
        }

        Py_ssize_t write = 0;
        if (value < 0.0) {
            for (Py_ssize_t index = 0; index < count; ++index) {
                Breakpoint item = breakpoints[index];
                if (item.is_top && item.point <= pivot) {
                    certified_alpha += item.alpha;
                    certified_beta += item.beta;
                }
                if (item.point > pivot) {
                    breakpoints[write++] = item;
                }
            }
        }
        else {
            for (Py_ssize_t index = 0; index < count; ++index) {
                Breakpoint item = breakpoints[index];
                if (!item.is_top && item.point >= pivot) {
                    certified_alpha += item.alpha;
                    certified_beta += item.beta;
                }
                if (item.point < pivot) {
                    breakpoints[write++] = item;
                }
            }
        }
        count = write;
    }

    if (certified_alpha == 0.0) {
        return smallest_top;
    }
    return certified_beta / certified_alpha;
}


static int
partial_sort_compute(
    const double *argument,
    double *result,
    Py_ssize_t dimension,
    Py_ssize_t selected,
    double gamma
)
{
    double safe_scale = 1.0 / sqrt(DBL_EPSILON);
    double maximum = 0.0;
    int has_positive = 0;
    for (Py_ssize_t index = 0; index < dimension; ++index) {
        double value = argument[index];
        if (!isfinite(value)) {
            return NATIVE_NONFINITE_ARGUMENT;
        }
        double magnitude = fabs(value);
        if (magnitude > maximum) {
            maximum = magnitude;
        }
        if (value > 0.0) {
            has_positive = 1;
        }
    }
    if (
        maximum > safe_scale
        || gamma > safe_scale
        || gamma < sqrt(DBL_MIN)
    ) {
        return NATIVE_UNSAFE_SCALE;
    }

    if (!has_positive) {
        memset(result, 0, (size_t)dimension * sizeof(double));
        return NATIVE_OK;
    }

    if (selected == dimension) {
        double denominator = 1.0 + gamma;
        for (Py_ssize_t index = 0; index < dimension; ++index) {
            double value = argument[index];
            if (value < 0.0) {
                value = 0.0;
            }
            value /= denominator;
            result[index] = value > 1.0 ? 1.0 : value;
        }
        return NATIVE_OK;
    }

    double *positive = malloc((size_t)dimension * sizeof(double));
    HeapItem *heap = malloc((size_t)selected * sizeof(HeapItem));
    unsigned char *top_mask = calloc((size_t)dimension, sizeof(unsigned char));
    double *top = malloc((size_t)selected * sizeof(double));
    Py_ssize_t *top_indices = malloc(
        (size_t)selected * sizeof(Py_ssize_t)
    );
    Py_ssize_t tail_size = dimension - selected;
    double *tail = malloc((size_t)tail_size * sizeof(double));
    double *top_breakpoints = malloc((size_t)selected * sizeof(double));
    Breakpoint *breakpoints = malloc(
        (size_t)dimension * sizeof(Breakpoint)
    );
    if (
        positive == NULL
        || heap == NULL
        || top_mask == NULL
        || top == NULL
        || top_indices == NULL
        || tail == NULL
        || top_breakpoints == NULL
        || breakpoints == NULL
    ) {
        free(positive);
        free(heap);
        free(top_mask);
        free(top);
        free(top_indices);
        free(tail);
        free(top_breakpoints);
        free(breakpoints);
        return NATIVE_MEMORY_ERROR;
    }

    for (Py_ssize_t index = 0; index < dimension; ++index) {
        positive[index] = argument[index] > 0.0 ? argument[index] : 0.0;
    }
    for (Py_ssize_t index = 0; index < selected; ++index) {
        heap[index].value = positive[index];
        heap[index].index = index;
    }
    for (Py_ssize_t index = selected / 2; index > 0; --index) {
        heap_sift_down(heap, selected, index - 1);
    }
    for (Py_ssize_t index = selected; index < dimension; ++index) {
        HeapItem candidate;
        candidate.value = positive[index];
        candidate.index = index;
        if (heap_item_is_better(&candidate, &heap[0])) {
            heap[0] = candidate;
            heap_sift_down(heap, selected, 0);
        }
    }
    for (Py_ssize_t index = 0; index < selected; ++index) {
        top[index] = heap[index].value;
        top_indices[index] = heap[index].index;
        top_mask[heap[index].index] = 1;
    }
    Py_ssize_t tail_position = 0;
    for (Py_ssize_t index = 0; index < dimension; ++index) {
        if (!top_mask[index]) {
            tail[tail_position++] = positive[index];
        }
    }

    double boundary = mixed_pool_boundary(
        top,
        selected,
        tail,
        tail_size,
        gamma,
        top_breakpoints,
        breakpoints
    );
    for (Py_ssize_t index = 0; index < dimension; ++index) {
        double value = positive[index] - boundary;
        if (value < 0.0) {
            value = 0.0;
        }
        if (value > 1.0) {
            value = 1.0;
        }
        result[index] = value;
    }
    for (Py_ssize_t index = 0; index < selected; ++index) {
        double subtraction = singleton_subtraction(top[index], gamma);
        if (boundary > subtraction) {
            subtraction = boundary;
        }
        double value = top[index] - subtraction;
        if (value < 0.0) {
            value = 0.0;
        }
        if (value > 1.0) {
            value = 1.0;
        }
        result[top_indices[index]] = value;
    }

    free(positive);
    free(heap);
    free(top_mask);
    free(top);
    free(top_indices);
    free(tail);
    free(top_breakpoints);
    free(breakpoints);
    return NATIVE_OK;
}


static PyObject *
native_partial_sort_into(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *argument_object = NULL;
    PyObject *result_object = NULL;
    double gamma = 0.0;
    Py_ssize_t selected = 0;
    if (!PyArg_ParseTuple(
        args,
        "OdnO:partial_sort_into",
        &argument_object,
        &gamma,
        &selected,
        &result_object
    )) {
        return NULL;
    }
    if (!isfinite(gamma) || gamma <= 0.0) {
        PyErr_SetString(
            PyExc_ValueError,
            "gamma must be positive and finite"
        );
        return NULL;
    }

    Py_buffer argument = {0};
    Py_buffer result = {0};
    int input_flags = PyBUF_FORMAT | PyBUF_C_CONTIGUOUS;
    int output_flags = input_flags | PyBUF_WRITABLE;
    if (PyObject_GetBuffer(argument_object, &argument, input_flags) < 0) {
        return NULL;
    }
    if (PyObject_GetBuffer(result_object, &result, output_flags) < 0) {
        PyBuffer_Release(&argument);
        return NULL;
    }

    int valid_buffers = (
        argument.ndim == 1
        && result.ndim == 1
        && argument.itemsize == (Py_ssize_t)sizeof(double)
        && result.itemsize == (Py_ssize_t)sizeof(double)
        && argument.format != NULL
        && result.format != NULL
        && strcmp(argument.format, "d") == 0
        && strcmp(result.format, "d") == 0
        && argument.len == result.len
        && argument.len > 0
    );
    if (!valid_buffers) {
        PyBuffer_Release(&result);
        PyBuffer_Release(&argument);
        PyErr_SetString(
            PyExc_ValueError,
            "argument and result must be equal-length contiguous Float64 vectors"
        );
        return NULL;
    }

    Py_ssize_t dimension = argument.len / (Py_ssize_t)sizeof(double);
    if (selected < 1 || selected > dimension) {
        PyBuffer_Release(&result);
        PyBuffer_Release(&argument);
        PyErr_SetString(
            PyExc_ValueError,
            "k must lie in {1, ..., dimension}"
        );
        return NULL;
    }

    int status = NATIVE_OK;
    Py_BEGIN_ALLOW_THREADS
    status = partial_sort_compute(
        (const double *)argument.buf,
        (double *)result.buf,
        dimension,
        selected,
        gamma
    );
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&result);
    PyBuffer_Release(&argument);

    if (status == NATIVE_MEMORY_ERROR) {
        return PyErr_NoMemory();
    }
    if (status == NATIVE_NONFINITE_ARGUMENT) {
        PyErr_SetString(PyExc_ValueError, "argument must be finite");
        return NULL;
    }
    if (status == NATIVE_UNSAFE_SCALE) {
        PyErr_SetString(
            PyExc_ValueError,
            "PAVA input is outside the conservative Float64 supported "
            "range; rescale the objective and data before calling the prox"
        );
        return NULL;
    }
    Py_RETURN_NONE;
}


static PyMethodDef native_methods[] = {
    {
        "partial_sort_into",
        native_partial_sort_into,
        METH_VARARGS,
        PyDoc_STR(
            "partial_sort_into(argument, gamma, k, result) -> None"
        )
    },
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef native_module = {
    PyModuleDef_HEAD_INIT,
    "_native_pava",
    "Native partial-selection PAVA kernels.",
    -1,
    native_methods,
    NULL,
    NULL,
    NULL,
    NULL
};


PyMODINIT_FUNC
PyInit__native_pava(void)
{
    return PyModule_Create(&native_module);
}
