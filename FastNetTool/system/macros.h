#ifndef FASTNETTOOL_SYSTEM_MACROS_H
#define FASTNETTOOL_SYSTEM_MACROS_H


#define OBJECT_SETTER_AND_GETTER(OBJ, TYPE, SETTER, GETTER)\
                                                            \
  TYPE GETTER(){                                            \
    return OBJ->GETTER();                                   \
  }                                                         \
  void SETTER(TYPE value){                                  \
    OBJ->SETTER(value);                                     \
    return;                                                 \
  }                                                         \
                                                            \

#define PRIMITIVE_SETTER_AND_GETTER(TYPE, SETTER, GETTER, VAR)\
                                                            \
  TYPE GETTER(){                                            \
    return VAR;                                             \
  }                                                         \
                                                            \
  void SETTER(TYPE value){                                  \
    VAR = value;                                            \
    return;                                                 \
  }                                                         \
                  
#define PRIMITIVE_SETTER(TYPE, SETTER,  VAR)                \
                                                            \
  void SETTER(TYPE value){                                  \
    VAR = value;                                            \
    return;                                                 \
  }                                                         \
                                                            \



#endif
