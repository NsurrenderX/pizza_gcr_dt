import numpy as np

def dump_ndarray_list(d:dict):
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            d[k] = v.tolist()
        elif isinstance(v, dict):
            d[k] = dump_ndarray_list(v)
    return d
def joint_equal(joint1:np.ndarray, joint2:np.ndarray):
        delta = np.abs(joint1 - joint2)
        if delta.max() > 0.000001:
            return False
        else:
            return True

if __name__ == '__main__':
    # a = {
    #     'a': np.array([1, 2, 3]),
    #     'b': {
    #         'c': np.array([4, 5, 6])
    #     },
    #     'd':{
    #         'e':{
    #             'f': {
    #                 'g': np.array([7, 8, 9])
    #             }
    #         }
    #     }
    # }
    # print(a)
    # a = dump_ndarray_list(a)
    # print(a)
    a = np.array([0.09979485, -0.45927223, -0.07632733, -2.62180612,  0.04726108,  2.30021298])
    b = np.array([0.09979485, -0.45927224, -0.07632733, -2.6218061,   0.04726108,  2.300213])
    delta = np.abs(a - b)
    print(delta.max())
    print(delta)