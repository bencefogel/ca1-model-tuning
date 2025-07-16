TITLE R-type calcium current (Cav2.3) Modified channel!

UNITS {
    (mV) = (millivolt)
    (mA) = (milliamp)
    (S) = (siemens)
}

NEURON {
    SUFFIX kslow
    USEION k READ ek WRITE ik
    RANGE gmax, ik
}

PARAMETER {
    gmax = 1 (S/cm2)
    ek (mV)
} 

ASSIGNED { 
    v (mV)
    ik (mA/cm2)
    celsius (degC)
    minf
    mtau (ms)
    }

STATE { m }

BREAKPOINT {
    SOLVE states METHOD cnexp
    ik = gmax*m*(v-ek)
    
}

INITIAL {
    rates()
    m = minf 
    ek = -75
}

DERIVATIVE states { 
    rates()
    m' = (minf-m)/mtau
}

PROCEDURE rates() {
    UNITSOFF
    minf = 1/(1+exp((v+15)/(-5)))
    mtau = 5+20/(1+exp((v+11)/(10)))
    UNITSON
}