import torch
from dataset import PoemDataset
from model import TransformerPoem


def generate_poem(model, dataset, start_text, max_len=50, temperature=1.0, device="cuda"):
    """生成诗歌"""
    model.eval()
    tokens = list(start_text)
    tokens = ["<bos>"] + tokens
    input_ids = [dataset.word2idx.get(token, dataset.word2idx["<unk>"]) for token in tokens]
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    generated_ids = input_ids
    for _ in range(max_len):
        with torch.no_grad():
            output = model(generated_ids, generated_ids)
            logits = output[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            if next_token.item() == dataset.word2idx["<eos>"]:
                break

    generated_tokens = [dataset.idx2word[idx.item()] for idx in generated_ids[0]]
    return "".join(generated_tokens[1:-1])  # 去掉<bos>和<eos>


def load_model_and_vocab(checkpoint_path, device="cuda"):
    """加载模型和词汇表"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    dataset = PoemDataset("archive/5yan.txt")
    model = TransformerPoem(
        vocab_size=len(dataset.word2idx), use_sparse_attn=("sparse" in checkpoint_path), use_glu=("glu" in checkpoint_path)
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, dataset


def main():
    # 加载模型和词汇表
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, dataset = load_model_and_vocab("ckpt/base_final.pt", device)
    print("欢迎使用古诗生成模型！输入前句以生成后续诗句，输入 'exit' 以退出。")
    print("=======================开始生成=======================")
    print()
    # 用户交互
    while True:
        start_text = input("请输入五言诗的前句：")
        if start_text.lower() in ["exit", "quit"]:
            print("程序退出。")
            break

        if len(start_text) != 5:
            print("请输入5个字的诗句~")
            continue

        # 生成诗歌
        generated_poem = generate_poem(model, dataset, start_text, device=device)
        print("生成的诗歌：", generated_poem)


if __name__ == "__main__":
    main()

