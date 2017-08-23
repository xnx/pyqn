import unittest
from ..symbol import Symbol

class SymbolCheck(unittest.TestCase):
    def test_symbol_init(self):
        s = Symbol()
        self.assertTrue(s.name == None)
        self.assertTrue(s.latex == None)
        self.assertTrue(s.html == None)
        self.assertTrue(s.definition == None)
        
        s = Symbol(name = 'NAME')
        self.assertTrue(s.name == 'NAME')
        self.assertTrue(s.latex == 'NAME')
        self.assertTrue(s.html == 'NAME')
        self.assertTrue(s.definition == None)
        
        s = Symbol(name = 'NAME', html = 'HTML')
        self.assertTrue(s.name == 'NAME')
        self.assertTrue(s.html == 'HTML')
        self.assertTrue(s.latex == 'NAME')
        self.assertTrue(s.definition == None)
        
        s = Symbol(definition = 'DEFINITION')
        self.assertTrue(s.name == None)
        self.assertTrue(s.html == None)
        self.assertTrue(s.latex == None)
        self.assertTrue(s.definition == 'DEFINITION')
        
        s = Symbol(name = 'NAME', latex = 'LATEX')
        self.assertTrue(s.name == 'NAME')
        self.assertTrue(s.html == 'NAME')
        self.assertTrue(s.latex == 'LATEX')
        self.assertTrue(s.definition == None)
        
    def test_symbol_str(self):
        self.assertTrue(str(Symbol()), '[undefined]')
        self.assertTrue(str(Symbol(name='NAME')),'NAME')
        self.assertTrue(str(Symbol(latex='LATEX')),'[undefined]')
        self.assertTrue(str(Symbol(name='NAME',latex='LATEX')),'NAME')

if __name__ == '__main__':
    unittest.main()

