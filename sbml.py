fun_dict = {}
var_stack = [{}]

class Node:
    def __init__(self):
        print("init node")

    def evaluate(self):
        return 0

    def execute(self):
        return 0

# Statements

class BlockNode(Node):
    def __init__(self, li):
        self.list = li

    def execute(self):
        for statement in self.list:
#            print('statement: {}'.format(statement))
            statement.execute()

class IDNode(Node):
    def __init__(self, val):
        self.val = val

    def evaluate(self):
        return self.val

class FuncNode(Node):
    def __init__(self, name, params, block, val):
        self.name = name.evaluate()
        self.params = []
        for param in params:
            self.params.append(param)
        self.block = block
        self.val = val
        fun_dict[self.name] = self

def TupleFullEval(node):
    temp = []
    for x in node:
        if isinstance(x, Node):
            x = x.evaluate()
        if type(x) is tuple:
            x = TupleFullEval(x)
#        print('x: {}'.format(x))
        temp.append(x)
#    print('result: {}'.format(temp))
    return tuple(temp)

class FuncCallNode(Node):
    def __init__(self, name, args):
        self.name = name.evaluate()
        self.args = args

    def evaluate(self):
        func = fun_dict[self.name]

        if len(self.args) != len(func.params):
            raise TypeError("Function {} takes {} arguments".format(func.name, len(func.params)))

        eval_args = {}
        for i in range(len(func.params)):
            eval_args[func.params[i].evaluate()] = self.args[i].evaluate()

        var_stack.append(eval_args)
        func.block.execute()
        val = func.val.evaluate()

        if isinstance(val, Node):
            val = val.evaluate()
        if type(val) is tuple:
            val = TupleFullEval(val)

        var_stack.pop()
#        print('Function {} evaluated to {}'.format(func.name, val))
        return val

    def execute(self):
        self.evaluate()

# VariableNode Helper Functions
def ListAssign(array, val, indices):
    i = indices[0].evaluate()
    if len(indices) == 1:
        array[i] = val
    else:
        array[i] = ListAssign(array[i], val, indices[1:])
    return array

def ListEval(array, indices):
    i = indices[0].evaluate()
    if len(indices) == 1:
        return array[i]
    else:
        return ListEval(array[i], indices[1:])

class VariableNode(Node):
    def __init__(self, var, index_list=None):
        self.var = var
        self.indices = index_list
        self.val = None

    def evaluate(self):
        #print('\tvar eval')
        key = self.var.evaluate()
        #print(key)
        #print(var_stack)
        val = var_stack[-1][key]
        if self.indices is None:
            return val
        else:
            return ListEval(val, self.indices)

    def assign(self, val):
        self.val = val

    def execute(self):
        if self.indices is None:
            key = self.var.evaluate()
            var_stack[-1][key] = self.val.evaluate()
        else:
            key = self.var.evaluate()
            array = var_stack[-1][key]
            var_stack[-1][key] = ListAssign(array, self.val.evaluate(), self.indices)

class IfNode(Node):
    def __init__(self, cond, block):
        self.cond = cond
        self.block = block

    def execute(self):
        if self.cond.evaluate():
            self.block.execute()

class IfElseNode(Node):
    def __init__(self, cond, block1, block2):
        self.cond = cond
        self.block1 = block1
        self.block2 = block2

    def execute(self):
        if self.cond.evaluate():
            self.block1.execute()
        else:
            self.block2.execute()

class WhileNode(Node):
    def __init__(self, cond, block):
        self.cond = cond
        self.block = block

    def execute(self):
        while self.cond.evaluate():
            self.block.execute()

class PrintNode(Node):
    def __init__(self, v):
        self.value = v

    def execute(self):
        output = self.value.evaluate()
        print(output)

# Expressions

class IntegerNode(Node):
    def __init__(self, v):
        self.value = int(v)

    def evaluate(self):
        return self.value

class RealNode(Node):
    def __init__(self, v):
        self.value = float(v)

    def evaluate(self):
        return self.value

class StringNode(Node):
    def __init__(self, v):
        v = v[1:-1]
        v = v.replace("\\'", "'")
        v = v.replace('\\"', '"')
        v = v.replace('\\\\', '\\')
        self.value = str(v)

    def evaluate(self):
        return self.value

class BoolNode(Node):
    def __init__(self, v):
        if v == 'True':
            self.value = True
        elif v == 'False':
            self.value = False
        else:
            raise TypeError('This is not a boolean')

    def evaluate(self):
        return self.value

class ListNode(Node):
    def __init__(self, v=None):
        if v is None:
            v = []
        self.value = v

    def add(self, v):
        self.value.insert(0,v)
        return self

    def part_eval(self):
        return self.value

    def evaluate(self):
        temp = []
        for v in self.value:
            temp.append(v.evaluate())
        return temp

class TupleNode(Node):
    def __init__(self, v=None):
        if v is None:
            v = ()
        self.value = v

    def add(self, v):
        self.value = (v,) + self.value
        return self

    def evaluate(self):
        return self.value

# Operators

class UMinusNode(Node):
    def __init__(self, v):
        self.value = -v

    def evaluate(self):
        return self.value

class UniOpNode(Node):
    def __init__(self, op, v):
        self.v = v
        self.op = op

    def evaluate(self):
        x = self.v.evaluate()
        if type(x) is bool:
            if (self.op == 'not'):
                return not x
        if type(x) is int or type(x) is float:
            if (self.op == '-'):
                return -x

        raise Exception("SEMANTIC ERROR")

def Cons(element, vector):
    temp = vector[:]
    temp.insert(0,element)
    return temp

class IndexNode(Node):
    def __init__(self, collection, index):
        self.coll = collection
        self.index = index

    def evaluate(self):
        e_coll = self.coll.evaluate()
        e_ind = self.index.evaluate()
        
        if (type(e_coll) is list or type(e_coll) is str):
            return e_coll[e_ind]
        elif (type(e_coll) is tuple):
            if isinstance(e_coll[e_ind-1], Node):
                return e_coll[e_ind-1].evaluate()
            else:
                return e_coll[e_ind-1]

        raise Exception("SEMANTIC ERROR")

class BinOpNode(Node):
    def __init__(self, op, v1, v2):
        self.v1 = v1
        self.v2 = v2
        self.op = op

    def evaluate(self):
        x = self.v1.evaluate()
        y = self.v2.evaluate()
        arg_type = type(x)

        if ((type(x) is int or type(x) is float) 
                and (type(y) is int or type(y) is float)):
            if (self.op == '+'):
                return x + y
            elif (self.op == '-'):
                return x - y
            elif (self.op == '*'):
                return x * y
            elif (self.op == '/'):
                return x / y
            elif (self.op == '**'):
                return x**y
        if (type(x) == type(y) or type(y) is list or type(y) is str):
            if (self.op == '::'):
                return Cons(x, y)
            if (type(y) is str or type(y) is list):
                if (self.op == 'in'):
                    return x in y
            if (arg_type is str or arg_type is list):
                if (self.op == '+'):
                    return x + y
                elif (self.op == '-'):
                    return x - y
            if (arg_type is int):
                if (self.op == 'div'):
                    return x // y
                elif (self.op == 'mod'):
                    return x % y
            if (arg_type is bool):
                if (self.op == 'andalso'):
                    return x and y
                elif (self.op == 'orelse'):
                    return x or y
        if (arg_type is int or arg_type is float or
                arg_type is str):
            if (self.op == '<>'):
                return x != y
            elif (self.op == '=='):
                return x == y
            elif (self.op == '<='):
                return x <= y
            elif (self.op == '>='):
                return x >= y
            elif (self.op == '<'):
                return x < y
            elif (self.op == '>'):
                return x > y

        # if no matches, then error
        raise Exception("SEMANTIC ERROR")

tokens = (
    'PRINT','IF','ELSE','WHILE','LCURLY','RCURLY',
    'ASSIGN','ID','FUNC',
    'SEMICOLON','LPAREN', 'RPAREN', 'LBRACKET', 'RBRACKET',
    'COMMA','TUPINDEX',
    'NUMBER','BOOL','STRING',
    'POWER',
    'TIMES','DIVIDE','QUOT_DIV','MODULO',
    'PLUS','MINUS',
    'IN','CONS',
    'NEQ','LEQ','REQ','EQ','LT','GT',
    'NOT','AND','OR'
    )

# Tokens
t_LCURLY  = r'{'
t_RCURLY  = r'}'

t_LPAREN  = r'\('
t_RPAREN  = r'\)'
t_POWER   = r'\*\*'
t_TIMES   = r'\*'
t_DIVIDE  = r'/'
t_PLUS    = r'\+'
t_MINUS   = r'-'

t_LBRACKET = r'\['
t_RBRACKET = r'\]'
t_COMMA    = r','
t_TUPINDEX = r'\#'
t_CONS    = r'::'
t_SEMICOLON = r';'

t_NEQ = r'<>'
t_LEQ = r'<='
t_REQ = r'>='
t_EQ  = r'=='
t_LT  = r'<'
t_GT  = r'>'

t_ASSIGN  = r'='

reserved = {
    'fun': 'FUNC',
    'if': 'IF',
    'else': 'ELSE',
    'while': 'WHILE',
    'print': 'PRINT',
    'in': 'IN',
    'not': 'NOT',
    'andalso': 'AND',
    'orelse': 'OR',
    'div': 'QUOT_DIV',
    'mod': 'MODULO',
}

def t_STRING(t):
    r'(\'(\\\'|[^\'])*\')|("(\\"|[^"])*")'
    try:
        t.value = StringNode(t.value)
    except:
        print("Unknown error while casting string")
        t.value = ""
    return t

def t_NUMBER(t):
    r'\d*(\d\.|\.\d)\d*(e-?\d+)? | \d+'
    try:
        if ('.' in t.value):
            t.value = RealNode(t.value)
        else:
            t.value = IntegerNode(t.value)
    except ValueError:
        print("Number value too large %d", t.value)
        t.value = 0
    return t

def t_BOOL(t):
    r'(True|False)'
    try:
        t.value = BoolNode(t.value)
    except TypeError:
        print("Not a properly formatted bool value")
        t.value = False
    return t

def t_ID(t):
    r'[A-Za-z][A-Za-z0-9_]*'
    try:
        t.type = reserved.get(t.value, 'ID')
        if t.type == 'ID':
            t.value = IDNode(t.value)
    except TypeError:
        print("Not a valid variable name")
        t.value = None
    return t

def t_error(t):
    raise Exception("SYNTAX ERROR")

# Ignored characters
t_ignore = " \t\n"
 
# Build the lexer
import ply.lex as lex
lex.lex(debug=0)

# Parsing rules
# low to high precedence
precedence = (
    ('right','ASSIGN'),
    ('left','OR'),
    ('left','AND'),
    ('left','NOT'),
    ('left','NEQ','LEQ','REQ','EQ','LT','GT'),
    ('right','CONS'),
    ('left','IN'),
    ('left','PLUS','MINUS'),
    ('left','TIMES','DIVIDE','QUOT_DIV','MODULO'),
    ('right','POWER'),
    ('right','UMINUS'),
    ('left','LBRACKET'),
    ('left','RBRACKET'),
    ('left','TUPINDEX'),
    ('left','COMMA'),
    ('left','LPAREN'),
    ('left','RPAREN'),
    ('nonassoc','IF'),
    ('nonassoc','ELSE'),
    ('nonassoc','WHILE', 'PRINT'),
    ('nonassoc','FUNC')
    )

# Program

def p_main(t):
    '''program : func_list block'''
    t[0] = t[2]

def p_func_list(t):
    '''func_list : func_list function
                 | function'''
    t[0] = None

def p_func_list_empty(t):
    '''func_list : '''
    t[0] = None

# Blocks

def p_block(t):
    '''block : LCURLY block_content RCURLY'''
    t[0] = BlockNode(t[2]);

def p_block_empty(t):
    '''block : LCURLY RCURLY'''
    t[0] = BlockNode([])

def p_block_content(t):
    '''block_content : block_content statement'''
    t[0] = t[1] + [t[2]]

def p_block_content_tail(t):
    '''block_content : statement'''
    t[0] = [t[1]]

# Statements

def p_statement(t):
    '''statement : block
                 | print_smt
                 | assignment
                 | if_smt
                 | if_else_smt
                 | while_smt'''
    t[0] = t[1]

def p_assignment(t):
    '''assignment : variable ASSIGN expression SEMICOLON'''
    t[0] = t[1]
    t[0].assign(t[3])

def p_variable(t):
    '''variable : ID'''
    t[0] = VariableNode(var=t[1])

def p_variable_list(t):
    '''variable : ID bracket_list'''
    t[0] = VariableNode(var=t[1], index_list=t[2])

def p_bracket_list(t):
    '''bracket_list : bracket_list LBRACKET expression RBRACKET'''
    t[1].append(t[3])
    t[0] = t[1]

def p_bracket_list_end(t):
    '''bracket_list : LBRACKET expression RBRACKET'''
    t[0] = [t[2]]

def p_print_smt(t):
    """
    print_smt : PRINT LPAREN expression RPAREN SEMICOLON
    """
    t[0] = PrintNode(t[3])

# Conditionals

def p_if_statement(t):
    ''' if_smt : IF LPAREN expression RPAREN block'''
    t[0] = IfNode(t[3], t[5])

def p_if_else_statement(t):
    ''' if_else_smt : IF LPAREN expression RPAREN block ELSE block'''
    t[0] = IfElseNode(t[3], t[5], t[7])

def p_while_statement(t):
    '''while_smt : WHILE LPAREN expression RPAREN block'''
    t[0] = WhileNode(t[3], t[5])

# Functions

def p_function(t):
    '''function : FUNC ID LPAREN param_list RPAREN ASSIGN block expression SEMICOLON'''
    t[0] = FuncNode(t[2], t[4], t[7], t[8])

def p_param_list(t):
    '''param_list : ID COMMA param_list'''
    t[3].insert(0,t[1])
    t[0] = t[3]

def p_param_single(t):
    '''param_list : ID'''
    t[0] = [t[1]]

def p_param_empty(t):
    '''param_list : '''
    t[0] = []

def p_function_call(t):
    '''factor : ID LPAREN arg_list RPAREN'''
    t[0] = FuncCallNode(name=t[1], args=t[3])

def p_function_call_stmt(t):
    '''statement : ID LPAREN arg_list RPAREN SEMICOLON'''
    t[0] = FuncCallNode(name=t[1], args=t[3])

def p_arg_list(t):
    '''arg_list : expression COMMA arg_list'''
    t[3].insert(0,t[1])
    t[0] = t[3]

def p_arg_single(t):
    '''arg_list : expression '''
    t[0] = [t[1]]

def p_arg_list_empty(t):
    '''arg_list : '''
    t[0] = []

# Expressions

def p_expression_parens(t):
    '''expression : LPAREN expression RPAREN'''
    t[0] = t[2]

def p_expression_binop(t):
    '''expression : expression PLUS expression
                  | expression MINUS expression
                  | expression TIMES expression
                  | expression DIVIDE expression
                  | expression QUOT_DIV expression
                  | expression MODULO expression
                  | expression POWER expression
                  | expression IN expression
                  | expression CONS expression
                  | expression NEQ expression
                  | expression LEQ expression
                  | expression REQ expression
                  | expression EQ expression
                  | expression LT expression
                  | expression GT expression
                  | expression AND expression
                  | expression OR expression'''
    t[0] = BinOpNode(t[2], t[1], t[3])

def p_expression_factor(t):
    '''expression : factor'''
    t[0] = t[1]

# Lists

def p_expression_list(t):
    '''factor : list'''
    t[0] = t[1]

def p_empty_list(t):
    '''list : LBRACKET RBRACKET'''
    t[0] = ListNode()

def p_list_tail_empty(t):
    '''list_tail : '''
    t[0] = ListNode()

def p_list_tail(t):
    '''list_tail : COMMA expression list_tail'''
    t[0] = t[3].add(t[2])

def p_list(t):
    '''list : LBRACKET expression list_tail RBRACKET'''
    t[0] = t[3].add(t[2])

def p_expression_list_index(t):
    '''factor : expression LBRACKET expression RBRACKET'''
    t[0] = IndexNode(t[1], t[3])

# Tuples

def p_expression_tuple(t):
    '''factor : tuple'''
    t[0] = t[1]

def p_empty_tuple(t):
    '''tuple : LPAREN RPAREN'''
    t[0] = TupleNode()

def p_tuple_tail_empty(t):
    '''tuple_tail : '''
    t[0] = TupleNode()

def p_tuple_tail(t):
    '''tuple_tail : COMMA expression tuple_tail'''
    t[0] = t[3].add(t[2])

def p_tuple(t):
    '''tuple : LPAREN expression COMMA expression tuple_tail RPAREN'''
    t[0] = t[5].add(t[4])
    t[0] = t[5].add(t[2])

# Other

def p_tuple_index(t):
    '''factor : TUPINDEX expression tuple'''
    t[0] = IndexNode(t[3], t[2])

def p_variable_tuple_index(t):
    '''factor : TUPINDEX expression variable'''
    t[0] = IndexNode(t[3], t[2])

def p_expression_tuple_index(t):
    '''factor : TUPINDEX expression LPAREN expression RPAREN'''
    t[0] = IndexNode(t[4], t[2])

def p_factor_literal(t):
    '''factor : NUMBER
              | BOOL
              | STRING'''
    t[0] = t[1]

def p_factor_variable(t):
    '''factor : variable'''
    t[0] = t[1]

def p_factor_uminus(t):
    '''expression : MINUS expression %prec UMINUS'''
    t[0] = UniOpNode(t[1], t[2])

def p_boolean_not(t):
    '''expression : NOT expression'''
    t[0] = UniOpNode(t[1], t[2])

def p_error(t):
    raise Exception("SYNTAX ERROR")

import ply.yacc as yacc
yacc.yacc(debug=0)

import sys

if (len(sys.argv) != 2):
    sys.exit("invalid arguments")
fd = open(sys.argv[1], 'r')
code = ""

for line in fd:
    code += line.strip()

try:
#if True:
    lex.input(code)
    while True:
        token = lex.token()
        if not token: break
        #print(token) # DEBUG

    ast = yacc.parse(code, debug=0)

    try:
    #if True:
        ast.execute()
    except Exception as e:
        print("SEMANTIC ERROR")
        #print(e) # DEBUG

except Exception as e:
    print("SYNTAX ERROR")
#    print(e) # DEBUG
