import unittest

if __name__ == '__main__':
    root_dir = './'
    loader = unittest.TestLoader()
    testSuite = loader.discover(start_dir='tests',
                                pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(testSuite)
