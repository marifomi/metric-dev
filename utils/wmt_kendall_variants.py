

variants_definitions = {

        'wmt12' : {
            '<' : { '<': 1 , '=':-1 , '>':-1  },
            '=' : { '<':'X', '=':'X', '>':'X' },
            '>' : { '<':-1 , '=':-1 , '>': 1  },
            },

        'wmt13' : {
            '<' : { '<': 1 , '=':'X', '>':-1  },
            '=' : { '<':'X', '=':'X', '>':'X' },
            '>' : { '<':-1 , '=':'X', '>': 1  },
            },

        'wmt14' : {
            '<' : { '<': 1 , '=': 0 , '>':-1  },
            '=' : { '<':'X', '=':'X', '>':'X' },
            '>' : { '<':-1 , '=': 0 , '>': 1  },
            },

        'xties' : {
            '<' : { '<': 1 , '=': 0 , '>':-1  },
            '=' : { '<': 0 , '=': 1 , '>': 0  },
            '>' : { '<':-1 , '=': 0 , '>': 1  },
            },

        }