from pdf import PDF
from linguistics.bert import BertVectorizer
from pandas import DataFrame
from pandas import concat
from chronometry.progress import ProgressBar
from disk import Path
from ravenclaw.wrangling import bring_to_front


class Bert:
	def __init__(self, num_tokens=50):
		self._bert_vectorizer = BertVectorizer(num_tokens=num_tokens)

	def vectorize(self, path, ignore_if_html_file_exists=True, ignore_errors=False):
		"""
		:type path: str or Path
		:type ignore_if_html_file_exists: bool
		:type ignore_errors: bool
		:rtype: DataFrame
		"""
		path = Path(path)
		pdf_paths = [
			PDF(file_path)
			for file_path in path.list(show_size=False)
			if file_path.extension.lower() == 'pdf'
		]

		exceptions = []

		# create htmls
		progress_bar = ProgressBar(total=len(pdf_paths))
		progress = 0
		num_fails = 0
		for pdf_path in pdf_paths:
			progress_bar.show(
				amount=progress,
				text=f'converting {pdf_path.name_and_extension} to html - errors: {num_fails}'
			)
			try:
				pdf_path.convert_to_html(ignore_if_exists=ignore_if_html_file_exists)
			except Exception as e:
				if ignore_errors:
					num_fails += 1
					exceptions.append(e)
				else:
					raise e
			progress += 1

		progress_bar.show(amount=progress, text=f'converting pdfs to html completed with {num_fails} errors.')

		# get paragraphs
		progress_bar = ProgressBar(total=len(pdf_paths))
		progress = 0
		num_fails = 0
		paragraphs = []
		for pdf_path in pdf_paths:
			progress_bar.show(
				amount=progress,
				text=f'extracting paragraphs of {pdf_path.name_and_extension} - errors: {num_fails}'
			)
			try:
				pdf_paragraphs = pdf_path.paragraphs
				num_paragraphs = len(pdf_paragraphs)
				for i, paragraph in enumerate(pdf_paragraphs):
					paragraphs.append({
						'pdf': pdf_path,
						'paragraph_num': i + 1,
						'paragraph': paragraph,
						'num_paragraphs': num_paragraphs
					})
			except Exception as e:
				if ignore_errors:
					num_fails += 1
					exceptions.append(e)
				else:
					raise e
			progress += 1

		progress_bar.show(amount=progress, text=f'extracting paragraphs completed with {num_fails} errors.')

		# create vectors
		progress_bar = ProgressBar(total=len(paragraphs))
		progress = 0
		num_fails = 0
		vectors = []
		for paragraph_dict in paragraphs:
			pdf_path = paragraph_dict['pdf']
			paragraph_num = paragraph_dict['paragraph_num']
			paragraph = paragraph_dict['paragraph']
			num_paragraphs = paragraph_dict['num_paragraphs']
			progress_bar.show(
				amount=progress,
				text=f'converting paragraphs of {pdf_path.name_and_extension} to vectors - errors: {num_fails}'
			)
			try:
				vector, num_tokens = self._bert_vectorizer.vectorize(text=paragraph, get_num_tokens=True)
				vector_df = DataFrame(
					vector,
					columns=[f'bert_{i + 1}' for i in range(vector.shape[1])]
				)
				vector_df['pdf'] = pdf_path.name_and_extension
				vector_df['num_paragraphs'] = num_paragraphs
				vector_df['paragraph_num'] = paragraph_num
				vector_df['num_tokens'] = num_tokens
				vectors.append(vector_df)
			except Exception as e:
				if ignore_errors:
					num_fails += 1
					exceptions.append(e)
				else:
					raise e
			progress += 1

		progress_bar.show(amount=progress, text=f'converting paragraphs to vectors completed with {num_fails} errors.')

		return bring_to_front(data=concat(vectors), columns=['pdf', 'paragraph_num', 'num_paragraphs', 'num_tokens'])
