TITLE R-type calcium current (Cav2.3) Modified channel!

UNITS {
    (mV) = (millivolt)
    (mA) = (milliamp)
    (S) = (siemens)
}

NEURON {
    SUFFIX car
    USEION ca READ cai, cao WRITE ica VALENCE 2
    RANGE gmax, ica
}

PARAMETER {
    gmax = 0.00021 (S/cm2)
} 

ASSIGNED { 
    v (mV)
    ica (mA/cm2)
    eca (mV)
    celsius (degC)
    cai (mM)
    cao (mM)
    minf
    mtau (ms)
    hinf
    htau (ms)
}

STATE { m h }

BREAKPOINT {
    SOLVE states METHOD cnexp
    ica = gmax*m*m*m*h*(v-eca)
    
}

INITIAL {
    rates()
    m = minf
    h = hinf
    eca = 80
}

DERIVATIVE states { 
    rates()
    m' = (minf-m)/mtau
    h' = (hinf-h)/htau
}

PROCEDURE rates() {
    UNITSOFF
    minf = 1/(1+exp((v+24)/(-15)))
    mtau = (2*v)/v
    hinf = 1/(1+exp((v+32)/(10)))
    htau = (100*v)/v
    UNITSON
}