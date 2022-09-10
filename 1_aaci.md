
# Search Techniques and Heuristics
```mermaid
flowchart LR
    A1[Solving Problems by Searching] --> A[8_puzzle_Problem] --> B[State Space] 
        B --> BA[Graph]
        B --> BB[Spanning Tree Algorithm]
            BB & BA--> BBA[Search as graph traversal]

                BBA --> BBAA[Uninformed Search]
                    BBAA --> BBAAA[Generic Search Algorithm]
                        BBAAA --> BBAAAA[Breadth First Search]
                            BBAAAA --- BBAAAAA["Complete"]
                            BBAAAA --- BBAAAAB["Not Optimal"]
                            BBAAAA --- BBAAAAC["Space Complexity: O(b^d)"]
                            BBAAAA --- BBAAAAD["Time Complexity: O(b^d)"]

                        BBAAA --> BBAAAB[Depth First Search] 
                            BBAAAB --- BBAAABA["Complete"]
                            BBAAAB --- BBAAABB["Not Optimal"]
                            BBAAAB --- BBAAABC["Space Complexity: O(mb)"]
                            BBAAAB --- BBAAABD["Time Complexity: O(b^d)"]
                            
                        BBAAA --> BBAAAC[Uniform Cost Search/UFS]  


                        BBAAA --> BBAAAD[Backtracking Search]
                            BBAAAD & BBAAGA --> BBAAADA[Min Max Search]
                                BBAAADA --- BBAAADAA[Alpha Beta Pruning] 

                        BBAAA --> BBAAAE[Iterative Deepening DFS] 
                            BBAAAE --- BBAAAEA["Complete"]
                            BBAAAE --- BBAAAEB["Optimal"]
                            BBAAAE --- BBAAAEC["Space Complexity: O(bd)"]
                            BBAAAE --- BBAAAED["Time Complexity: O(b^d) "]

                        BBAAA --> BBAAAF[Bidirectional Search]
                        BBAAA --> BBAAG[Adversial Search]
                            BBAAGA[Game Problem Formulation]--> BBAAG

                BBA --> BBAB[Informed Search]
                    BBAB --> BBABA[Best First Search]
                        BBABA --> BBABAA[Uniform Cost Search]
                            BBABAA --- BBABAAA["Complete"]
                            BBABAA --- BBABAAB["Optimal"]
                            BBABAA --- BBABAAC["Space Complexity: O(b^m)"]
                            BBABAA --- BBABAAD["Time Complexity: O(b^(1+floor(c*/ε)))"]

                        BBABA --> BBABAB[Greedy Best First Search]
                            BBABAB --- BBABABA["Not Complete"]
                            BBABAB --- BBABABB["Not Optimal"]
                            BBABAB --- BBABABC["Space Complexity: O(b^m) "]
                            BBABAB --- BBABABD["Time Complexity: O(b^m)"]

                        BBABA --> BBABAC[A* Search]
                        BBABA --> BBABAD[Weighted A* Search]
                        BBABA --> BBABAE[Beam Search]
                        BBABA --> BBABAEF[Iterative Deepening A* Search]
                    BBAB --> BBABB[Heuristics]
                        BBABB --- BBABBA[Admissiable]
                        BBABB --- BBABBB[Consistent]
```

<div class = "d-flex flex-row justify-content-center">
    <img src ="C:\Users\Gurudeep\Desktop\Deep Learning Book\AACI\test1.jpg" alt = "drawing" width="700"/><br>
    <img src ="C:\Users\Gurudeep\Desktop\Deep Learning Book\AACI\test2.jpg" alt = "drawing" width="500"/><br>
</div>

|Methods|Complete(1/0)|Optimal(1/0)|Space Complexity|Time Complexity|
|:---|:----:|:---:|:---:|:----:|
| BFS|1|0|$O(b^d)$|$O(b^d)$|
| DFS|1|0|$O(mb)$|$O(b^d)$|
| UFS|1|1|$O(mb)$|$O(b^{1+floor(\frac{C^*}{\epsilon})})$|
| Bidirectional Search|1|-|$O(bd)$|$O(b^{\frac{d}{2}})$|
| Greedy Best First Search|0|0|$O(b^m)$|$O(b^m)$|
| $A^*$ Search|1|1|-|-|
| Min Max Algorithm|1|1|$O(mb)$|$O(b^m)$|




# Game Theory
```mermaid
flowchart LR
A[Game Theory] --> AA[MinMax  Search] --- AAA[Alpha Beta Pruning]
```
# Constraint Satisfaction Problem
```mermaid
flowchart LR
    C1[Search in Complex Environments] --> C[Possible World] --> CA[Variable]
        CA --> CAA[Domain]
            CAA --> CAAA[Constraint]
                CAAA --> CAAAA[Constraint Satisfaction Problem]
                    CAAAA --> CAAAAA[Soduku as CSP]
                    CAAAA --> CAAAAB[Task Scheduling as CSP]
                    CAAAA --> CAAAAC[Map Colouring as CSP]
                    CAAAA --> CAAAAD[Crypto Arithmetic as CSP]
                    CAAAAA & CAAAAB & CAAAAC & CAAAAD --> CAAAAE[Solution for CSP]
                        CAAAAE --> CAAAAEA[CSP as Search Problem]
                            CAAAAEA --> CAAAAEAA[Back Tracking Search]
                                CAAAAEAA --> CAAAAEAAAA[MRV-Minimum Remaining Values/Most Constraining Value ]
                                CAAAAEAAAA --> CAAAAEAAAB[Forward Checking]
                                    CAAAAEAAAB --> CAAAAEAAABA[Constraint Propagation: Heuristic]
                                    CAAAAEAAABA --> CAAAAEAAABAA[Arc Consistency]
                                    CAAAAEAAABA --> CAAAAEAAABAB[Path Consistency]
                    CAAAA --> CAAAAF[Varieties of CSP]
                        CAAAAF --> CAAAAFA[Discrete variable]
                            CAAAAFA --> CAAAAFAA[Finite Domain]
                            CAAAAFA --> CAAAAFAB[Infinite Domain]
                        CAAAAF --> CAAAAFB[Continous Variables]



                CAAA --> CAAAB[Variety of Constraints]
                    CAAAB --> CAAABA[Unary Constraint]
                    CAAAB --> CAAABB[Binary Constraint]
                    CAAAB --> CAAABC[Higher Order Constraint]
                    CAAAB --> CAAABD["Preference(Soft Constraints)"] 
                        CAAABD --- CAAABDA[Constraint Optimisation Problem]
```

# Logic and AI
```mermaid
graph LR;
A[Knowledge Base] --> AA[Sematics]
A --> AB[Tautologies]
A --> AC[Logical Equivalences]
A --> AD[Inference]
A --> AE[Valid Arguments]
    AE --> AEA[Proof Method]
        AEA --> AFA
A --> AF[Propositional Logic]
    AF --> AFA[Rules of Inference]
        AFA --> AFAA[Modus Ponens]
        AFA --> AFAB[Modus Tollens]
    AF --> AFB[Clauses]
        AFB --> AFBA[CNF-Conjunctive Normal Form]
            AFBA --> AFBAA[Resolution for CNF]
            AFBA --> AFBAB[Resolution Special Cases]
            AFBA --> AFBAC[Tree Proof]
            AFBA --> AFBAD[Resolution Refutation Proof]
            AFBA --> AFBAE[Proof By Contradiction]
            AFBA --> AFBAF[Conversion to CNF]
        AFB --> AFBB[DNF-Disjunctive Normal Form]
    AF --> AFC[Unsatisfiable Formula]
    AF --> AFD[Unconstrained SAT]
    AF --> AFE[Subsumption]
    AF --> AFF[Satisfiablility Problems]
    AF --> AFG[2-SAT]
        AFG --> AFGA[Walk SAT Algo]
    AF --> AFH[Horn Clause]
A --> AG[First Order Logic]
    AG --> AGA[Syntax of FOL]
        AGA --> AGAA[Constant Symbols]
        AGA --> AGAB[Predicate Symbols]
        AGA --> AGAC[Function Symbols]
        AGA --> AGAD[Arity of Predicate/Function]
    AG --> AGB[Component of FOL]
        AGB --> AGBA[Term]
            AGBA --> AGBAA[Constant Symbols]
            AGBA --> AGBAB[Function Symbols]
        AGB --> AGBB[Atomic Sentence]
            AGAB & AGBA -.- AGBB
        AGB --> AGBC[Complex Sentence]
    AG --> AGC[Quantifiers]
        AGC --> AGCA[Universal Quantifier]
        AGC --> AGCB[Existential Quantifier]
    AG --> AGD[Reasoning with FOL]
        AGD --> AGDA[Generalisation]
            AGDA --> AGDAA[Universal Generalisation]
            AGDA --> AGDAB[Existential Generalisation]
        AGD --> AGDB[Instantiation]
            AGDB --> AGDBA[Universal Instantiation]
            AGDB --> AGDBB[Existential Instantiation]
        AGD --> AGDC[Inference]
            AGDC --> AGDCA[Resolution]
            AGDC --> AGDCB[Substitution]
            AGDC --> AGDCC[Unification]
                AGDCC --- AGDCCA[Most General Unifier]
            AGDC --> AGDCD[Skolemisation]
            AGDC --> AGDCE[Forward Chaining]
            AGDC --> AGDCF[Backward Chaining]
            AGDC --> AGDCG[Resolution-Refutation]
            AGDC --> AGDCH[Equality]

```
# Ai And Planning

```mermaid
graph LR;
A[AI Planning] --> AB[Planning Domain] 
    AB -.- AA[Described by using on relational representations using predicate and objects using FOL]
    AB --> ABA[Classical Planning Problem]
        ABA --- ABAA[Given Domain D]
            ABAA --> ABAAA[Initial State]
                ABAAA --- ABAAAA[Predicate completion is applied to it]
                ABAAA --- ABAAAB[Domain Closure is applied to it]
                ABAAA --- ABAAAC[Unique name assumption is applied to it]
            ABAA --> ABAAB[Goal State]
            ABAA --> ABAAC[D]
        ABA --- ABAB[State Space]
            ABAB --> ABABA[All States]
            ABAB --> ABABB[Actions as operators]
                ABABB -.- ABABBA[Actions are deterministic]
        ABA --- ABAC[Output]
            ABAC -.- ABACA[Sequence of actions that will lead to goal state]
        ABABA & ABAAB & ABABB --> ABACB[Representation]
            ABACB --- ABACBA[Representing States]
                ABACBA -.- ABACBAA[Closed World Assumption]
            ABACB --- ABACBB[Representing Goals]
            ABACB --- ABACBC[Representing Action in Strips]
                ABACBC --> ABACBCA[Set of PRE condtition facts, ADD, DEL]
                ABACBC --> ABACBCB[Sematics for Strip Actions]
                ABACBC --> ABACBCC[Strips Planning Problem]
                ABACBC --> ABACBCD[Strips Action Schemas]
    AB --> ABB[Planning Domain Description Language PDDL]
    
    AB --> ABC[Properties of Planners]
        ABC --- ABCA[Sound]
        ABC --- ABCB[Complete]
        ABC --- ABCC[Optimal]

    AB --> ABD[Planning as Graph Search]
        ABD --- ABDA[Nodes -> Possible States]
        ABD --- ABDB[Directed Arcs -> STRIPS actions]
        ABD --- ABDC[Solution]
        ABD ---  ABDD[Graph Plan]
            ABDD --> ABDDA[Graph Expansion]
            ABDD --> ABDZDB[Solution Extraction]-->ABDDB[Constructing Planning Graph]
                ABDDB --> ABDDBA[Mutual Exclusive Relations-Mutex]
                    ABDDBA --> ABDDBAA[Actions being Mutex] 
                        ABDDBAA --- ABDDBAAA[Inconsistent Effects]
                        ABDDBAA --- ABDDBAAB[Interference]
                        ABDDBAA --- ABDDBAAC[Competing needs]
                    ABDDBA --> ABDDBAB[Propositions Being Mutex]
                        ABDDBAB --- ABDDBABA[If one is negation of other]
                        ABDDBAB --- ABDDBABB[Inconsistent support]
                    ABDDBAAA & ABDDBAAB & ABDDBAAC & ABDDBABA & ABDDBABB --> ABDE[Graph Plan algorithm]
            ABDE --> ABDEA[BackTrack Search for Solution Extraction]
            ABDE -.- ABDEB[Its Polynomial Time]
                ABDEB -.- ABDEBA["Max nodes proposition level O(p+mln^k)"]
                ABDEB -.- ABDEBB["Max nodes action level: O(mn^k)"]

B[Techniques to solve Classical Planning Problem] --> BA[Progession/Forward Search]
B --> BB[Regression/Backward Search]
B --> BC[SAT Based Planners]
B --> BD[Graph Plan-> Sir Taught]
```

# Fuzzy Logic

# Evolutionary Algorithms
```mermaid
graph LR;
C[(Biologically Inspired Algorithms)]--> CA(Evolutionary Algorithms)
C --> CB(Swarm Algorithms)
C --> CC(Fuzzy Logic)
C --> CD(Artificial Neural Networks)
```
```mermaid
graph LR;
A[Evolutionary Algorithms] -->B[From Natural Selection of Species] --> BA[through natural selection of chromosomes] --> BAA[to Natural Selection of Optimal Solutions]
```
```mermaid
graph LR;
A(Evolutionary Algorithms) --> AA(N-dimensional Solution space)
    AA --- AAA(one point that is global optimum)
    AA --- AAB(objective function-> global min/max)
    AA --- AAC(every point associated using fitness fn)
        AAC --- AACA(1-1 mapping bet loc of each point & its fitness)
    AA --> AAD(initial guess <=> no.of canditate soln called population)
        AAD --> AADA(Traverse total soln space across generations)
            AADA --> AADAA(Evolve to higher and higher levels of fitness)
                AADAA --- AADAAA(Changes state i.e values of its component variables)

A --> B[Differential Evolution]
    B --> BA(Generate New Solution)
        BA --> BAA(Create Mutant vector)
            BAA --> BAAA(Use Mutant Vector as cross over partner for old soln to gen new soln->trail vector)
            BAAA --> BAAAA(If new soln has better fitness than parent ,replace parent with new soln )

A --> C(Particle Swarm Optimisation)

A --> D(Classical Genetic Algorithms)
    D --- DA(Mutation)
        DA --> DAA(Selection)
            DAA --- DAAA(Fitness fn eval for all points)
                DAAA --- DAAAA(Tournament Selection)
                DAAA --- DAAAB(Roulette Wheel Selection)
        DAAAA & DAAAB --> DAB[Mutation,crossover -> Elitism]
```
# Genetic Programming

```mermaid
graph LR;
A(Genetic Algorithms) --> B(Representation of Software in Classical GP)
A --> C(Primitive Sets)
    C --> CA(Function Set)
        CA --- CAA(Nodes - function set)
            CAA -.- CAAA(Combines subtrees)
        CA --- CAB(Leaves - terminal set)
A --> D(Fitness of Programs)
A --> E(Steps to implement it)
    E --> EA(Initialisation)
    EA --> EB(Crossover)
    EB --> EC(Mutation)
        EC --- ECA(Sub Tree Mutation)
        EC --- ECB(Point Mutation)
    ECA & ECB --> ED(Selection)
        ED --- EDA(Fitness Proportionate Selection-Roulette Wheel)
        ED --- EDB(Tournement Selection)
    EDA & EDB --> EE(Closure and Suffiency)
    EE --> EF(Parameter Set)
A --> F(Current Impetus)

```
























