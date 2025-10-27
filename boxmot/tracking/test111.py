class ABC(object):
    def __init__(self):  # 1. 补充 self 参数
        pass
    
    # @staticmethod
    def AA(method):
        # 2. 实现装饰器逻辑：返回原方法或包装后的方法
        def wrapper(*args, **kwargs):
            # 这里可以添加装饰器的额外逻辑（如日志、验证等）
            print("装饰器 AA 被调用")
            return method(*args, **kwargs)  # 调用原方法
        return wrapper  # 返回包装函数
    
    @AA  # 现在 AA 是一个有效的装饰器
    def BB(self, args):
        print(f"BB 方法被调用，参数：{args}")


aa = ABC()
aa.BB('test')