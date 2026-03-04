import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print("="*60)
print("测试HydroMTL_CGC所有必要依赖")
print("="*60)

# 测试基础科学计算包
packages = [
    ("numpy", "np"),
    ("pandas", "pd"),
    ("xarray", "xr"),
    ("torch", "torch"),
    ("yaml", "yaml"),
    ("sklearn", "sklearn"),
    ("matplotlib", "matplotlib"),
    ("scipy", "scipy"),
]

for module_name, short_name in packages:
    try:
        if module_name == "yaml":
            import yaml
            version = "N/A"
        elif module_name == "sklearn":
            import sklearn
            version = sklearn.__version__
        else:
            exec(f"import {module_name} as {short_name}")
            version = eval(f"{short_name}.__version__")
        
        print(f"✅ {module_name:15} {version}")
    except ImportError as e:
        print(f"❌ {module_name:15} 未安装")
    except Exception as e:
        print(f"⚠️  {module_name:15} 导入错误: {str(e)[:50]}")

print("\n" + "="*60)
print("测试项目特定模块...")

# 测试项目内部模块
project_modules = [
    "mtl_cgc.utils.config_parser",
    "mtl_cgc.utils.logger",
    "mtl_cgc.data.loader",
    "mtl_cgc.models.mtl_model",
    "mtl_cgc.training.trainer",
    "mtl_cgc.evaluation.evaluator",
]

for module in project_modules:
    try:
        __import__(module)
        print(f"✅ {module}")
    except ImportError as e:
        print(f"❌ {module}: {str(e)[:100]}")
    except Exception as e:
        print(f"⚠️  {module}: {str(e)[:100]}")

print("\n" + "="*60)
print("所有依赖测试完成！")
print("="*60)
