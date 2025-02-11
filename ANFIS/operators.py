import numpy as np


def productN(args, op):
    return np.prod(args, axis=0)


def zadehT(args, op):
    return np.min(args, axis=0)


def zadehS(args, op):
    return np.max(args, axis=0)


def algebraicT(args, op):
    return np.prod(args, axis=0)


def probabilisticS(args, op):
    return np.sum(args, axis=0) - np.prod(args, axis=0)


def lukasiewiczT(args, op):
    return np.max((np.sum(args, axis=0) - 1.0, np.zeros_like(np.sum(args, axis=0) - 1.0)), axis=0)


def lukasiewiczS(args, op):
    return np.min((np.sum(args, axis=0), np.ones_like(np.sum(args, axis=0))), axis=0)


def einsteinT(args, op):
    nominator = np.prod(args, axis=0)
    denominator = np.full_like(nominator, 2.0) - np.sum(args, axis=0) - nominator
    return nominator / denominator


def einsteinS(args, op):
    nominator = np.sum(args, axis=0)
    denominator = np.ones_like(nominator) + np.prod(args, axis=0)
    return nominator / denominator


def fodorT(args, op):
    return np.where(np.sum(args, axis=0) > 1, np.min(args, axis=0), np.zeros_like(np.sum(args, axis=0)))


def fodorS(args, op):
    return np.where(np.sum(args, axis=0) < 1, np.max(args, axis=0), np.ones_like(np.sum(args, axis=0)))


def drasticT(args, op):
    return np.where(np.max(args, axis=0) == 1, np.min(args, axis=0), np.zeros_like(np.max(args, axis=0)))


def drasticS(args, op):
    return np.where(np.min(args, axis=0) == 0, np.max(args, axis=0), np.ones_like(np.min(args, axis=0)))
