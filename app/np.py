import math
import gc
import cmath
import random as urandom
def conj(a):
    return a.real-1j*a.imag
class Vector(object):
    def __init__(self, *args):
        """ Create a vector, example: v = Vector(1,2) """
        if len(args)==0: self.values = (0,0)
        else: self.values = args
    def norm(self):
        """ Returns the norm (length, magnitude) of the vector """
        return math.sqrt(sum( comp**2 for comp in self ))
    def cnorm(self):
        return math.sqrt(sum((comp*conj(comp)).real for comp in self))
    def _zero_mean(self):
        mean = self.mean()
        zeroed = tuple(i-mean for i in self)
        return zeroed
    def zero_mean(self):
        return Vector(*self._zero_mean())
    def mean(self):
        return sum(self)/len(self)      
    def argument(self):
        """ Returns the argument of the vector, the angle clockwise from +y."""
        arg_in_rad = math.acos(Vector(0,1)*self/self.norm())
        arg_in_deg = math.degrees(arg_in_rad)
        if self.values[0]<0: return 360 - arg_in_deg
        else: return arg_in_deg

    def normalize(self):
        """ Returns a normalized unit vector """
        norm = self.norm()
        normed = tuple( comp/norm for comp in self )
        return Vector(*normed)
    def zero_mean_normalize(self):
        zeroed = self.zero_mean()
        return zeroed.normalize()
    def rotate(self, *args):
        """ Rotate this vector. If passed a number, assumes this is a 
            2D vector and rotates by the passed value in degrees.  Otherwise,
            assumes the passed value is a list acting as a matrix which rotates the vector.
        """
        if len(args)==1 and type(args[0]) == type(1) or type(args[0]) == type(1.):
            # So, if rotate is passed an int or a float...
            if len(self) != 2:
                raise ValueError("Rotation axis not defined for greater than 2D vector")
            return self._rotate2D(*args)
        elif len(args)==1:
            matrix = args[0]
            if not all(len(row) == len(v) for row in matrix) or not len(matrix)==len(self):
                raise ValueError("Rotation matrix must be square and same dimensions as vector")
            return self.matrix_mult(matrix)
        
    def _rotate2D(self, theta):
        """ Rotate this vector by theta in degrees.
            
            Returns a new vector.
        """
        theta = math.radians(theta)
        # Just applying the 2D rotation matrix
        dc, ds = math.cos(theta), math.sin(theta)
        x, y = self.values
        x, y = dc*x - ds*y, ds*x + dc*y
        return Vector(x, y)  
    def gen_matrix_mult(self, matrix):
        """ Multiply this vector by a matrix.  Assuming matrix is a list of lists.
        
            Example:
            mat = generator of lists
            Vector(1,2,3).matrix_mult(mat) ->  (14, 2, 26)
         
        """
        # Grab a row from the matrix, make it a Vector, take the dot product, 
        # and store it as the first component
        product = len(self)*[0.00]
        for idx,row in enumerate(matrix):
            product[idx] = Vector(*row)*self
            try:
                gc.collect()
            except:
                pass
        return Vector(*product)
    def matrix_mult(self, matrix):
        """ Multiply this vector by a matrix.  Assuming matrix is a list of lists.
        
            Example:
            mat = [[1,2,3],[-1,0,1],[3,4,5]]
            Vector(1,2,3).matrix_mult(mat) ->  (14, 2, 26)
         
        """
        """if not all(len(row) == len(self) for row in matrix):
           raise ValueError('Matrix must match vector dimensions') 
        """
        # Grab a row from the matrix, make it a Vector, take the dot product, 
        # and store it as the first component
        product = tuple(Vector(*row)*self for row in matrix)
        return Vector(*product)
    
    def inner(self, other):
        """ Returns the dot product (inner product) of self and other vector
        """
        return sum(a * b for a, b in zip(self, other))
    
    def __mul__(self, other):
        """ Returns the dot product of self and other if multiplied
            by another Vector.  If multiplied by an int or float,
            multiplies each component by other.
        """
        if type(other) == type(self):
            return self.inner(other)
        elif type(other) == type(1) or type(other) == type(1.0):
            product = tuple( a * other for a in self )
            return Vector(*product)
    
    def __rmul__(self, other):
        """ Called if 4*self for instance """
        return self.__mul__(other)
            
    def __div__(self, other):
        if type(other) == type(1) or type(other) == type(1.0):
            divided = tuple( a / other for a in self )
            return Vector(*divided)
    
    def __add__(self, other):
        """ Returns the vector addition of self and other """
        added = tuple( a + b for a, b in zip(self, other) )
        return Vector(*added)
    
    def __sub__(self, other):
        """ Returns the vector difference of self and other """
        subbed = tuple( a - b for a, b in zip(self, other) )
        return Vector(*subbed)
    def __pow__(self, other):
        powed = tuple(a**other for a in self)
        return Vector(*powed)
    
    def __iter__(self):
        return iter(self.values)#.__iter__()
    
    def __len__(self):
        return len(self.values)
    
    def __getitem__(self, key):
        return self.values[key]
        
    def __repr__(self):
        return str(self.values)
def change_idx(vector, idx, new_value):
    copied = list(vector)
    copied[idx] = new_value
    return Vector(*copied)

def get_column(nested_lst, idx):
    return [i[idx] for i in nested_lst]

def get_column_as_vec(nested_lst, idx):
    return Vector(*get_column(nested_lst, idx))

def radix2(x):
    N = len(x)
    if N <= 1: return x
    even = radix2(x[0::2])
    odd =  radix2(x[1::2])
    T= [cmath.exp(-2j*cmath.pi*k/N)*odd[k] for k in range(N//2)]
    return [even[k] + T[k] for k in range(N//2)] + \
           [even[k] - T[k] for k in range(N//2)]
def zero_mean(x):
    a = Vector(*x)
    return a._zero_mean()
def fft(x):
    return radix2(list(zero_mean(x)))
def rand_unif():
    return 1-2*urandom.getrandbits(8)/255
def spectral_mat(ws):
    one_row = lambda i,lst: [i*conj(e) for e in lst]
    all_rows = lambda lst:[one_row(i,lst) for i in lst]
    return all_rows(ws)
def pagerank(lst_of_lists, max_iter = 100):
    n = len(lst_of_lists)
    initial = Vector(*[rand_unif()+1j*rand_unif() for i in range(n)])
    for n in range(max_iter):
        new_initial = Vector(*initial)
        xi=new_initial.matrix_mult(lst_of_lists)
        l_norm = xi.cnorm()
        xi = [i/l_norm for i in xi]
        if (new_initial-xi).cnorm()<1e-10:
            break
    return xi







