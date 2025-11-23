from ..tensor import Tensor

class Module:
    def parameters(self):
        for attr in self.__dict__.values():
            if isinstance(attr, Tensor) and attr.requires_grad:
                yield attr
            elif isinstance(attr, Module):
                yield from attr.parameters()
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, Module):
                        yield from item.parameters()
                    elif isinstance(item, Tensor) and item.requires_grad:
                        yield item

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)