---
layout: page_project
title: CS784 - Foundations of Data Management
description: Formalizing Foundations of Data Management Results using Proof Assistants
img: assets/img/projects/cs784/miniatura.png
importance: 5
category: work
github: https://github.com/andreyyao/DBLean4

---

## Abstract

In this project, we focus on formalizing key results in database management theory using the **_Lean4_** proof assistant. Our primary objective is to verify the correctness of theorems that characterize query containment under different semantics. Specifically we formalize: 1) traditional set semantic notions regarding query containment, like the Homomorphism theorem, 2) notions regarding Union of Conjunctive Queries (UCQ), 3) containment results regarding semiring semantics found in [Green, 2009][1]. During this formalization process we discovered a falacy in one of the theorems from [Green, 2009][1] concerning the mapping relationship between different semirings and query containment.

## Introduction

### Semiring Semantics
Semiring semantics is a generalization of ordinary set semantics that unifies important settings like incomplete databases, probabilistic databases, bag semantics and why-provenance. This observation originated from the seminal work of [GTK, 2007][2] and influenced greatly the field of Foundations of Data Management. In this section we will introduce the necessary background on set semantics needed to motivate and understand the project.

A **semiring** is a set equipped with two operations:
- __*Addition*__ (+): A commutative, associative operation with identity 0. 
- __*Multiplication*__ ($$\cdot$$): An associative operation with identity 1, distributing over addition.

Additionally, \(0\) is an absorbing element for multiplication: 0 $$\cdot$$ a = a $$\cdot$$ 0 = 0. In semiring semantics we annotate the facts of the database using elements of a semiring $$(K,+,\cdot, 0,1)$$ and not just true or false values.
This gives us a generalization of the typical boolean setting where annotation provides additional information about a element of the database,
This additional information can refer to confidence scores, costs, access level and not just the validity of the facts. Leading to more complex but useful queries like Weighted Path finding, Information Retrieval on documents ranked by relevance and queries with aggregation of likelihoods.

There are many examples of semirings that can give rise to interesting semantics. 
From practical perspective some of the most useful ones are [Gradel, 2023][3]:

- The *Boolean* semiring $$\mathbb{B}=(\{0,1\},\lor,\land,0,1)$$ gives us boolean query evaluation, i.e., asking if there exists a tuple satisfying specific conditions in the database.
- The *Natural number* semiring $$\mathbb{N}=(\mathbb{N},+,\cdot,0,1)$$ gives us bag semantics, i.e., asking how many tuples satisfy specific conditions in the database.
- The *Natural coefficient polynomial* semiring $$(\mathbb{N}[X],+,\cdot,0,1)$$ gives us provenance polynomials, i.e., asking which combinations of the facts are responsible for a given statement.
- The *Tropical* semiring $$\mathbb{T} = (\mathbb{R}_{+}^{\infty}, \min, +, \infty, 0)$$ for cost interpretations.
- The *Viterbi* semiring $$\mathbb{V} = ([0, 1], \max, \cdot, 0, 1)$$ for confidence scores.

We will only consider commutative semirings that are naturally ordered by addition:
$$a \leq_{K} b \; \Leftrightarrow \; \exists c\in K \, (a + c = b)$$

##### Definition: Query Evaluation
The evaluation of a query $$Q$$ of the form $$Q(\vec{u}) :- R_1(\vec{u}_1), \ldots, R_n(\vec{u}_n) $$ on an instance $$I$$ with respect to $$K$$ semantics is given by $$Q(I) = \lambda t . \sum_{\nu \text{ s.t. } \nu(\vec{Q}) = t} \prod_{R_i\in Q}^{} R_i(\nu(\vec{u}_i))$$ where $$\lambda t$$ represents the mapping of tuples to values in $$K$$, and the sum and product operations are defined in $$K$$.

##### Definition: Query Containment
Let $$K$$ be a naturally-ordered semiring and let $$R_1, R_2$$ be two relations. We define containment of $$R_1$$ in $$R_2$$ by
$$R_1 \leq_K R_2 \quad  \quad \Leftrightarrow \quad \forall t \, R_1(t) \leq R_2(t)$$. We define containment of queries $$P, Q$$ with respect to $$K$$ semantics by $$P \sqsubseteq_K Q \quad  \quad \Leftrightarrow \quad \forall I \, P(I) \leq_K Q(I)$$

The main statements that we will be concerned with regarding semiring semantics in our formalization work are:

##### Proposition A.2 (Green, 2009)
Let $$K_1$$, $$K_2$$ be naturally-ordered commutative semirings. If $$h:K_1\to K_2$$ is a semiring homomorphism then for all $$a,b\in K_1$$, $$a\leq_{K_1} b \implies h(a)\leq_{K_2} h(b)$$. If $$h$$ is also surjective, then for all $$a,b\in K_1$$, $$a\leq_{K_1} b \iff h(a) \leq_{K_2} h(b)$$

##### Lemma 6.2 (Green, 2009)
For naturally-ordered semirings $$K_1, K_2$$, if there exists a surjective homomorphism $$h : K_1 \to K_2 $$, then $$K_1 \Rightarrow K_2$$.

These are statements that connect the existance of a homomorphism relationship to different semirings and query containment  under semiring semantics.

__*Remark*__: We note that the statement of Proposition A2 is wrong as stated. Specifically the second part of the statement that draws additional conclusions when the homomorphism is also subjective is faulty. This was discoverd during the formalization procedure and we refer the reader to the corresponding Section for the details.

### Proof assistants 
Interactive Proof assistants are programming languages built for writing computer-checked proofs in an interactive way. Notable applications using these tools include proving four color theorem and verifying a C compiler. Modern proof assistants like [Lean4][4] and Coq are quite interpretable and host large libraries of mathematical structures like semirings. 


As an example, see listing below for a basic proof in Lean4 to one of the building bock statements formalized: Given a CQ $$q$$: $$\textbf{head}(q) \in q (D[q])$$

```lean
/-- Define the canonical database `D[q]` of a query `[q]` -/
def canonical_DB (q : @CQ S V outs) : @Instance S V :=
  fun (R : S.relSym) =>
  { t : @Vect (S.arities R) V | { R := R , vars := t } ∈ q.body }

/-- The head of any query `q` is an element of `q(D[q])`-/
lemma head_in_canon_db_eval : ∀ (q : @CQ S V outs), q.head ∈ semantics q (canonical_DB q) := by
  intro q
  unfold semantics canonical_DB
  rw [Set.mem_setOf_eq]
  exists id;
  apply And.intro
  . rw [Vect.map_id]
  . intro A mem; rw [Vect.map_id, Set.mem_setOf_eq]
    exact mem
```

Lean4 also provides a context window, which keeps track of available assumptions and the current goal. Code 1 below shows the context of previous example program.

    Tactic state
    1 goal
    case right
    S : Schema
    outs : ℕ
    V : Type
    q : CQ V outs
    A : Atom S V
    mem : A ∈ q.body
    ⊢ { R := A.R, vars := A.vars } ∈ q.body

## Motivation in verifying results for UCQs under semiring semantics
The main motivation in this project is that simple results regarding containment (Homomorphism theorem) do not extend over semiring semantics. 

Because of the importance of understanding how relations between different semirings affect query containment, here we formalize some theorems regarding semiring semantics of UCQs using a proof assistant. In particular, we employ Lean4 over other proof assistants available like Coq, Isabelle, or Agda because of its mature mathematical library for abstract algebra (Mathlib.Algebra).

Some of the contributions of this work include:

- Relations, Instances, and Syntax of CQ/UCQ.
- Definitions of set semantics, semiring semantics, and containment for the semantics, resp.
- The homomorphism theorem for set semantics.
- The containment lemma (6.2) from [Green, 2009][1] for semiring semantics and lemmas leading up to it.
        

## Formalized Results

One of the benefits of using proof assistants is that they automatically verify the correctness of
proofs via type-checking. This means that once a proof is formalized, it is guaranteed to be
correct, requiring no further testing. However, it is important to make sure that the specific formal
encodings of the definitions and theorem statements themselves make sense. To ensure this we also provided informal justifications for important definitions and theorem statements.

### Bulding block structures

Specifically, the building block structures for the data management definitions used throughout this work are as follows:

- __Schema__ as a collection of relational symbols, each associated with an specific arity.
```lean
structure Schema where
relSym : Type /-Relation symbol -/
arities : relSym -> Nat /-Num vars -/
```

- __Atoms__ as a relational symbol along with variables.
```lean
def Atom.map (f : V1 -> V2) 
   (A : Atom S V1) : Atom S V2 :=
{ R := A.R, vars := Vect.map f A.vars }
```

- __Instance__ as a function that assigns each relational symbol to a set of tuples with arity matching the relational symbol’s specification.
```lean
def Instance (D : Type) := 
    Π (R : S.relSym), 
    Set (@Vect (S.arities R) D)
```

- __Conjunctive query (CQ)__ as a structure of a vector of variables (denoting the head) and a list of Atoms (denoting the body).
```lean
structure CQ (V : Type) 
             (outs : Nat) where
    head : @Vect outs V
    body : List (@Atom S V)
```

- __Homomorphism__ as a mapping between two conjunctive queries that preserves the structure of the query by mapping variables in the head and body accordingly.
```lean
structure homomorphism {V1 V2 : Type} (q1 : @CQ S V1 outs) 
    (q2 : @CQ S V2 outs) 
    (h : V1 -> V2) : Prop where
body_cond : ∀ A ∈ q1.body, 
            (Atom.map h A) ∈ q2.body
head_cond : q2.head = Vect.map h q1.head
```

- __Semantics__ of a conjunctive query as the set of output tuples derived from variable assignments satisfying the head and body constraints in a given instance.
```lean
def semantics {D : Type} 
    (q : @CQ S V outs) 
    (I : @Instance S D) : 
    Set (@Vect outs D) :=
{ t : Vect D |
    ∃ v : V -> D,
    t = Vect.map v q.head /\
    ∀ A ∈ q.body, (Vect.map v A.vars) ∈ (I A.R) }
```

- __Canonical Database__ of a query as an instance where each relational symbol is associated with tuples derived directly from the query’s body.
```lean
def canonical_DB (q : @CQ S V outs) : @Instance S V :=
fun (R : S.relSym) =>
  { t : @Vect (S.arities R) V | { R := R , vars := t } ∈ q.body }
```

- We proved the __Homomorphism Theorem__ by providing a lemma for each implication.
```lean
theorem homomorpshim_theorem :
  (q1.head ∈ semantics q2 (canonical_DB q1) ↔ contained V1 q1 q2) ∧
  (contained V1 q1 q2 ↔ ∃ h, homomorphism q2 q1 h) := by
  apply And.intro;
  . apply Iff.intro
    . intro H; apply homomorphism_2_1; apply homomorphism_3_2; apply H
    . apply homomorphism_1_3
  . apply Iff.intro
    . intro H; apply homomorphism_3_2; apply homomorphism_1_3; apply H
    . intro H; apply homomorphism_2_1; exact H
```

- __UCQ__ as a list of conjunctive queries (CQs) with the same variable set and output arity.
```lean
def UCQ V outs := List (@CQ S V outs)
```

- __Set Semantics of a UCQ__ as the set of output tuples that satisfy at least one of the CQs in a given instance.
```lean
def set_semantics (qs : @UCQ S V outs) (I : @Instance S D) : 
    Set (@Vect outs D) :=
  { t |
    ∃ q ∈ qs,
    ∃ v : V -> D,
    Vect.map v q.head = t /\
    ∀ A ∈ q.body, (Vect.map v A.vars) ∈ (I A.R) }
```

- __Semiring Semantics of a UCQ__ as the cumulative result over all CQs, computed in a specified semiring, for each output tuple.
```lean
def CQ_semiring_semantics 
(q : @CQ S V outs) (I : @Instance S D K) 
(t : @Vect outs D) : K :=
let valuations := { v : V -> D | Vect.map v q.head = t }
let valuations' := Set.Finite.toFinset (finite_impl_finite_set valuations)
∑ v ∈ valuations', List.foldl (fun (acc : K) (A : Atom S V) => acc * (I A.R (Vect.map v A.vars))) 1 q.body
```

- __Tuple__ as a structure containing a relational symbol and its associated values, representing an entry in a database.
```lean
structure tuple where
R : S.relSym
val : @Vect (S.arities R) adom
```

- __Valuation__ as a mapping of variables to domain elements, allowing the evaluation of variable-based queries in specific contexts.
```lean
variable {V : Type} [Fintype V]
variable {D : Type} [Fintype D]
def valuation := V -> D
```

- __Natural Order__ as a partial ordering on a semiring, defined such that one element is less than or equal to another if a third element can be added to the first to obtain the second.
```lean
def natural_order (K : Type) 
    [Semiring K] : K -> K -> Prop :=
    fun (a b : K) => ∃ (c : K), a + c =b
```


- __Boolean Semiring__ by interpreting operations such as addition and multiplication in the boolean context.
```lean
instance Bool.instSemiring : CommSemiring Bool where
  add := or
  add_assoc := by intros; exact Bool.or_assoc _ _ _
  zero := false
  zero_add := by intros; exact Bool.false_or _
  add_zero := by intros; exact Bool.or_false _
  add_comm := by intros; exact Bool.or_comm _ _
  mul := and
  mul_assoc := by intros; exact Bool.and_assoc _ _ _
  mul_comm := by intros; exact Bool.and_comm _ _
  one := true
  one_mul := by intros; exact Bool.true_and _
  mul_one := by intros; exact Bool.and_true _
  left_distrib := by intros; exact Bool.and_or_distrib_left _ _ _
  right_distrib := by intros; exact Bool.and_or_distrib_right _ _ _
  zero_mul := by intros; exact Bool.false_and _
  mul_zero := by intros; exact Bool.and_false _
  nsmul := Bool.nsmul
  nsmul_zero := by intro b; rfl
  nsmul_succ := by intros n b; simp; rw [Bool.or_comm]; rfl
```

- __Annotation of an Instance with Boolean values__ as an interpretation of an instance that maps each relation to a Boolean indicating the presence of a tuple, used for Boolean set semantics.
```lean
def annotate_with_bool : 
    @Instance S D Bool :=
fun (R : S.relSym) 
    (t : @Vect (S.arities R) D) => 
    t ∈ I R
```

### Proposition, Lemmas, Theorems involving semiring semantics

For the general case of semiring semantics, one of the first formalized propositions state that any function $$h$$ can be used to transform $$K-$$relations into $$K'-$$relations simply by applying $$h$$ to each tagged-tuple relations.

##### Proposition 6.1 (Green, 2009)
Let $$h:K\to K'$$ and assume that $$K, K'$$ are commutative semirings. Then $$\bar{Q}(h(I)) = h(\bar{Q}(I))$$ for all $$\bar{Q}\in UCQ$$ and $$K$$-instances $$I$$ iff $$h$$ is a semiring homomorphism.

```lean
lemma homomorphism_semantics_commute 
    {Q : @UCQ S V outs} 
    {I : @Instance S D K1} 
    {hom : RingHom K1 K2} :
semiring_semantics Q (Instance.map hom I) = 
KRel.map hom (semiring_semantics Q I) := by
unfold semiring_semantics KRel.map; simp; funext t
induction Q with
| nil => simp
| cons hd tl _ =>
simp; rw [homomorphism_semantics_commute_CQ]; simp; rw [List.foldl_hom]
intros k A; rw [map_add]
rw [<- homomorphism_KRel_map_commute, homomorphism_semantics_commute_CQ]
```

Many different semirings ($$\mathbb{B}, \mathbb{N}, \mathcal{P}(\mathcal{P}(X)), \dots$$) can be used for annotations. More importantly, the existence of surjective homomorphisms (structure-preserving maps) relating semirings determine properties such as query containment (and therefore also equivalence).

The following lemma formalizes this behavior by stating that for all UCQs $$\bar{Q_1}, \bar{Q_2}$$, if $$\bar{Q_1} \sqsubseteq_{K_1} \bar{Q_2}$$ then $$\bar{Q_1} \sqsubseteq_{K_2} \bar{Q_2}$$.

##### Lemma 6.2 (Green, 2009)
For naturally-ordered semirings $$K_1, K_2$$, if there exists a surjective homomorphism $$h:K_1\to K_2$$, then $$K_1 \Rightarrow K_2$$. 

```lean
lemma epimorphism_imp_determines {D : Type} [Fintype D] (K1 K2 : Type) [NatOrdSemiring K1] [NatOrdSemiring K2]
(hom : RingHom K1 K2) 
(surj : Function.Surjective hom)
  : @K_determines outs V1 _ V2 _ D _ S K1 K2 _ _ := by
  unfold K_determines UCQ_semiring_contains KRel.contained;
  intros Q1 Q2 query_contains; intros J t
  let I : @Instance S D K1 :=
    fun R (t : @Vect (S.arities R) D) => Function.surjInv surj (J R t)
  have map_I_J : J = Instance.map hom I := by
    unfold Instance.map; funext R t; unfold I; rw [Function.surjInv_eq surj]
  specialize query_contains I t
  rw [map_I_J]
  rw [homomorphism_semantics_commute, homomorphism_KRel_map_commute]
  rw [homomorphism_semantics_commute, homomorphism_KRel_map_commute]
  apply homomorphism_monotone; exact query_contains
```

Finally, we formalized relationships concerning containment/equivalence of UCQs for arbitrary semirings:

##### Theorem 6.4 (Green, 2009)
For all $$K$$, $$\mathbb{N}[X] \Rightarrow K$$. For all positive $$K$$, $$K \Rightarrow \mathbb{B}$$.

We have proved the second half of the statement, while the first half is still underdevelopment

### A use case for formalization: Fallacy discovered in a published result on database queries with semiring annotations
During the formalization process, a subtle flaw was found in one of the lemmas presented in proposition A.2 of [Green, 2009][1]. Specifically, the proof assistant flagged an issue in the second part of the statement, preventing it from proceeding due to a gap in the reasoning. The backward direction of Proposition A.2. (highlighted in red) is wrong. This can be shown by counterexample, for instance by taking $$K_2$$ as the trivial semiring $$K_2=0$$. For this claim to hold it is sufficient that $$h$$ is an isomorphism. Fortunately, the rest of the paper did not seem to depend on the second half of proposition A.2.

##### Proposition A.2 [Green, 2009][1]
Let $$K_1$$, $$K_2$$ be naturally-ordered commutative semirings. If $$h:K_1\to K_2$$ is a semiring homomorphism then for all $$a,b\in K_1$$, $$a\leq_{K_1} b \implies h(a)\leq_{K_2} h(b)$$. If $$h$$ is also surjective, then for all $$a,b\in K_1$$, $$a\leq_{K_1} b \iff h(a) \leq_{K_2} h(b)$$.


## Conclusions

Formalization plays an important role in ensuring the correctness of the theory of database queries with semiring annotations. Interactive proof assistants have become increasingly more powerful and usable in recent years, and can help rigorously check these operations automatically. In this project, we used the Lean4 proof assistant to rigorously verify different results in foundations of database management, particularly on the relationships between queries under different semantics.

Our formalization process, starting from foundational concepts like set semantics and the homomorphism theorem and extending to advanced results on the semiring semantics of union of conjunctive queries (UCQs), demonstrated the value of formal methods. The discovery of a flaw in one of these results already published underscores the importance of rigorous verification and the role of formalization in refining and improving existing theoretical work.

## References
1. [Containment of conjunctive queries on annotated relations, Green et. al. 2009](https://dl.acm.org/doi/10.1145/1514894.1514930)
2. [Provenance semirings, Green et. al. 2007](https://dl.acm.org/doi/10.1145/1265530.1265535)
3. [Semiring Semantics, Grädel 2023](https://simons.berkeley.edu/sites/default/files/2023-12/LADT23-3%20Slides%20-%20Erich%20Graedel.pdf)
4. [The Lean 4 Theorem Prover and Programming Language, De Moura, 2021](https://dl.acm.org/doi/10.1007/978-3-030-79876-5_37)

-----

[1]: https://dl.acm.org/doi/10.1145/1514894.1514930
[2]: https://dl.acm.org/doi/10.1145/1265530.1265535
[3]: https://simons.berkeley.edu/sites/default/files/2023-12/LADT23-3%20Slides%20-%20Erich%20Graedel.pdf
[4]: https://dl.acm.org/doi/10.1007/978-3-030-79876-5_37