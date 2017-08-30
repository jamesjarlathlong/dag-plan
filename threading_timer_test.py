import time
import multiprocessing as mp
import functools
def timeit(method):
    def timed(*args, **kw):
        ts = time.clock()
        result = method(*args, **kw)
        te = time.clock()
        ex_time = te-ts
        print('ex time: ', method, ex_time)
        return ex_time,result
    return time

def foo_pool(x):
    time.sleep(2)
    print('got here')
    return x*x

def comp(f, *args,**kw):
	ts = time.clock()
	result = f(*args, **kw)
	print('got here too')
	te = time.clock()
	ex_time = te-ts
	print('ex time: ', f, ex_time)
	return ex_time,result

t_f = functools.partial(comp, foo_pool)
result_list = []
def log_result(result):
    # This is called whenever foo_pool(i) returns a result.
    # result_list is modified only by the main process, not the pool workers.
    result_list.append(result)

def apply_async_with_callback():
    pool = mp.Pool()
    g = pool.apply_async(t_f, args = (1, ), callback = log_result)
    pool.close()
    pool.join()
    print(result_list, g)

if __name__ == '__main__':
    apply_async_with_callback()
