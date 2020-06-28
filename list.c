/*-----------------------------------------------------------------*/
/*
 Implantation du TAD List vu en cours.
 */
/*-----------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "list.h"

typedef struct s_LinkedElement {
	int value;
	struct s_LinkedElement *previous;
	struct s_LinkedElement *next;
} LinkedElement;

/* Use of a sentinel for implementing the list :
 The sentinel is a LinkedElement * whose next pointer refer always to the head of the list and previous pointer to the tail of the list
 */
struct s_List {
	LinkedElement *sentinel;
	int size;
};

typedef struct s_SubList {
	LinkedElement *head;
	LinkedElement *tail;
	int size;
}SubList;

/*-----------------------------------------------------------------*/

List *list_create() {
	List *l = malloc(sizeof(struct s_List));
	l->sentinel = malloc(sizeof(struct s_LinkedElement));
	l->sentinel->previous = l->sentinel->next = l->sentinel;
	l->size = 0;
	return l;
}

/*-----------------------------------------------------------------*/

List *list_push_back(List *l, int v) {
	LinkedElement *e = malloc(sizeof(struct s_LinkedElement));
	e->value = v;
	e->next = l->sentinel;
	e->previous = l->sentinel->previous;
	e->previous->next = e;
	l->sentinel->previous = e;
	++(l->size);
	return l;
}

/*-----------------------------------------------------------------*/

void list_delete(ptrList *l) {
	for(LinkedElement *e = (*l)->sentinel->next; e != (*l)->sentinel; e = e->next){
		free(e);
	}
	(*l) = NULL;
}

/*-----------------------------------------------------------------*/

List *list_push_front(List *l, int v) {
	LinkedElement *e = malloc(sizeof(struct s_LinkedElement));
	e->value = v;
	e->previous = l->sentinel;
	e->next = l->sentinel->next;
	e->next->previous = e;
	l->sentinel->next = e;
	++(l->size);
	return l;
}

/*-----------------------------------------------------------------*/

int list_front(List *l) {
	return (l->sentinel->next->value);
}

/*-----------------------------------------------------------------*/

int list_back(List *l) {
	return (l->sentinel->previous->value);
}

/*-----------------------------------------------------------------*/

List *list_pop_front(List *l) {
	LinkedElement *e = l->sentinel->next;
	l->sentinel->next = e->next;
	e->next->previous = l->sentinel;
	--(l->size);
	free(e);
	return l;
}

/*-----------------------------------------------------------------*/

List *list_pop_back(List *l){
	LinkedElement *e = l->sentinel->previous;
	l->sentinel->previous = e->previous;
	e->previous->next = l->sentinel;
	--(l->size);
	free(e);
	return l;
}

/*-----------------------------------------------------------------*/

List *list_insert_at(List *l, int p, int v) {
	assert(p >= 0 && p <= l->size);
	LinkedElement *at = l->sentinel->next;
	for(; p != 0; --p, at = at->next);
	LinkedElement *e = malloc(sizeof(struct s_LinkedElement));
	e->value = v;
	e->next = at;
	e->previous = at->previous;
	e->next->previous = e;
	e->previous->next = e;
	++(l->size);
	return l;
}

/*-----------------------------------------------------------------*/

List *list_remove_at(List *l, int p) {
	assert(p >= 0 && p < l->size);
	LinkedElement *remove = l->sentinel->next;
	while(p--){
		remove = remove->next;
	}
	remove->previous->next = remove->next;
	remove->next->previous = remove->previous;
	free(remove);
	--(l->size);
	return l;
}

/*-----------------------------------------------------------------*/

int list_at(List *l, int p) {
	assert(p >= 0 && p <= l->size);
	LinkedElement *at = l->sentinel->next;
	for(; p != 0; --p, at = at->next);
	return (at->value);
}

/*-----------------------------------------------------------------*/

bool list_is_empty(List *l) {
	return (list_size(l) == 0);
}

/*-----------------------------------------------------------------*/

int list_size(List *l) {
	return (l->size);
}

/*-----------------------------------------------------------------*/

List * list_map(List *l, SimpleFunctor f) {
	for(LinkedElement *e = l->sentinel->next; e != l->sentinel; e = e->next){
		e->value = f(e->value);
	}
	return l;
}


List *list_reduce(List *l, ReduceFunctor f, void *userData) {
	for(LinkedElement *e = l->sentinel->next; e != l->sentinel; e = e->next){
		f(e->value, userData);
	}
	return l;
}

/*-----------------------------------------------------------------*/


SubList list_split(SubList l){
	assert(l.size > 1);
	SubList res;
	LinkedElement *e = l.head;
	for(int i = 1; i != l.size/2; i++){
		e = e->next;
	}
	res.head = e;
	res.tail = e->next;
	return res;
}

SubList list_merge(SubList leftList, SubList rightList, OrderFunctor f){
	SubList l;

	/* Initialisation */
	if(f(leftList.head->value, rightList.head->value)){
		l.head = leftList.head;
		if(leftList.size > 1){
			leftList.head = leftList.head->next;
			--(leftList.size);
		}
	}else{
		l.head = rightList.head;
		if(leftList.size > 1){
			rightList.head = rightList.head->next;
			--(rightList.size);
		}
	}
	l.tail = l.head;

	/* Parcours */
	while(l.tail != leftList.tail && l.tail != rightList.tail){
		if(f(leftList.head->value, rightList.head->value)){
			leftList.head->previous = l.tail;
			l.tail->next = leftList.head;
			l.tail = leftList.head;
			leftList.head = leftList.head->next;
			--(leftList.size);
		}else{
			rightList.head->previous = l.tail;
			l.tail->next = rightList.head;
			l.tail = rightList.head;
			rightList.head = rightList.head->next;
			--(rightList.size);
		}
	}

	/* Rajout des derniers elements */
	if(l.tail != leftList.tail){
		leftList.head->previous = l.tail;
		l.tail->next = leftList.head;
		l.tail = leftList.tail;
	}else{
		rightList.head->previous = l.tail;
		l.tail->next = rightList.head;
		l.tail = rightList.tail;
	}

	l.size = leftList.size + rightList.size;

	return l;
}
	
SubList list_mergesort(SubList l, OrderFunctor f){
	SubList leftList;
	SubList rightList;
	SubList merge;
	SubList new_l;
	if(l.size <= 1){
		return l;
	}else{
		new_l = list_split(l);
		leftList.tail = new_l.head;
		leftList.head = l.head;
		rightList.head = new_l.tail;
		rightList.tail = l.tail;
		leftList.size = 1;
		for(LinkedElement *e = leftList.head; e != leftList.tail; e = e->next){
			++(leftList.size);
		}
		rightList.size = l.size - leftList.size;
		leftList = list_mergesort(leftList, f);
		rightList = list_mergesort(rightList, f);
		merge = list_merge(leftList, rightList, f);
		merge.size = leftList.size + rightList.size;
		return merge;
	}
}

List *list_sort(List *l, OrderFunctor f){
	SubList subL;
	List *res = list_create();
	res->size = l->size;
	subL.head = l->sentinel->next;
	subL.tail = l->sentinel->previous;
	subL.size = l->size;
	subL = list_mergesort(subL, f);
	subL.head->previous = res->sentinel;
	subL.tail->next = res->sentinel;
	res->sentinel->next = subL.head;
	res->sentinel->previous = subL.tail;
	return res;
}

